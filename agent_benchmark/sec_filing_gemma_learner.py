"""Deterministic linear two-head learner for the SEC/Gemma experiment.

The module performs no file, network, clock, market-data, SEC, or model I/O.
It implements the one frozen ridge-logistic probability head and ridge-Huber
edge head used by both the semantic predictor and its no-semantics ablation.
All persisted floating-point values use canonical ``float.hex`` strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import re
from typing import Any, Final

import numpy as np

from agent_benchmark.sec_filing_gemma_contract import canonical_sha256


MODEL_TYPE: Final[str] = "aapl_sec_filing_gemma_linear_two_head_v1"
STATE_SCHEMA_VERSION: Final[int] = 1
HEAD_VARIANTS: Final[frozenset[str]] = frozenset({"semantic", "ablation"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class SecFilingGemmaLearnerError(ValueError):
    """Raised when learner inputs or state violate the frozen contract."""


@dataclass(frozen=True)
class SecFilingGemmaLearnerConfig:
    raw_mad_multiplier: float = 1.4826
    raw_scale_floor: float = 1e-6
    raw_z_clip: float = 4.0
    ridge_lambda: float = 0.1
    logistic_max_iterations: int = 50
    logistic_tolerance: float = 1e-10
    newton_line_search_max_steps: int = 50
    huber_delta: float = 1.5
    huber_max_iterations: int = 50
    huber_tolerance: float = 1e-10
    target_mad_multiplier: float = 1.4826
    target_scale_floor: float = 1e-6
    edge_clip_lower: float = -0.5
    edge_clip_upper: float = 0.5

    def validate(self) -> None:
        observed = asdict(self)
        expected = asdict(SecFilingGemmaLearnerConfig())
        if set(observed) != set(expected):
            raise SecFilingGemmaLearnerError(
                "Learner configuration differs from the frozen SEC/Gemma v1 design"
            )
        for name, expected_value in expected.items():
            value = observed[name]
            if isinstance(expected_value, int):
                matches = (
                    not isinstance(value, bool)
                    and isinstance(value, int)
                    and value == expected_value
                )
            else:
                matches = (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float, np.number))
                    and math.isfinite(float(value))
                    and float(value).hex() == float(expected_value).hex()
                )
            if not matches:
                raise SecFilingGemmaLearnerError(
                    "Learner configuration differs from the frozen SEC/Gemma v1 design"
                )


@dataclass(frozen=True)
class SecFilingGemmaFitMetadata:
    candidate_sha256: str
    head_variant: str
    fold_id: str
    train_label_maturity_through: str
    training_set_sha256: str
    training_row_count: int
    maximum_training_label_maturity_session: str
    feature_schema_sha256: str

    @classmethod
    def coerce(
        cls, value: Mapping[str, Any] | "SecFilingGemmaFitMetadata"
    ) -> "SecFilingGemmaFitMetadata":
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            if set(value) != set(cls.__dataclass_fields__):
                raise SecFilingGemmaLearnerError("Fit metadata has invalid keys")
            try:
                result = cls(**dict(value))
            except TypeError as exc:
                raise SecFilingGemmaLearnerError("Fit metadata is invalid") from exc
        else:
            raise SecFilingGemmaLearnerError("Fit metadata must be a mapping")
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
            "candidate_sha256",
            "training_set_sha256",
            "feature_schema_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.head_variant not in HEAD_VARIANTS:
            raise SecFilingGemmaLearnerError("Unknown learner head variant")
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise SecFilingGemmaLearnerError("fold_id must be nonblank")
        from datetime import date

        try:
            cutoff = date.fromisoformat(self.train_label_maturity_through)
            maximum = date.fromisoformat(
                self.maximum_training_label_maturity_session
            )
        except (TypeError, ValueError) as exc:
            raise SecFilingGemmaLearnerError(
                "Fit metadata dates must be canonical ISO dates"
            ) from exc
        if (
            cutoff.isoformat() != self.train_label_maturity_through
            or maximum.isoformat()
            != self.maximum_training_label_maturity_session
            or maximum > cutoff
        ):
            raise SecFilingGemmaLearnerError(
                "Training label maturity exceeds the frozen fold cutoff"
            )
        if (
            isinstance(self.training_row_count, bool)
            or not isinstance(self.training_row_count, int)
            or self.training_row_count < 2
        ):
            raise SecFilingGemmaLearnerError(
                "training_row_count must be an integer of at least two"
            )


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaLearnerError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _float_hex(value: Any, location: str = "float") -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise SecFilingGemmaLearnerError(f"{location} must be a finite float")
    number = float(value)
    if not math.isfinite(number):
        raise SecFilingGemmaLearnerError(f"{location} must be a finite float")
    return number.hex()


def _decode_float_hex(value: Any, location: str) -> float:
    if not isinstance(value, str):
        raise SecFilingGemmaLearnerError(
            f"{location} must be a canonical hexadecimal float"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecFilingGemmaLearnerError(
            f"{location} must be a canonical hexadecimal float"
        ) from exc
    if not math.isfinite(number) or number.hex() != value:
        raise SecFilingGemmaLearnerError(
            f"{location} must be a canonical finite hexadecimal float"
        )
    return number


def _config_state(config: SecFilingGemmaLearnerConfig) -> dict[str, Any]:
    """Encode the frozen config without persisting decimal JSON floats."""

    config.validate()
    result: dict[str, Any] = {}
    for name, value in asdict(config).items():
        if isinstance(value, int):
            result[name] = value
        else:
            result[name] = _float_hex(value, f"config.{name}")
    return result


def _config_from_state(value: Any) -> SecFilingGemmaLearnerConfig:
    defaults = asdict(SecFilingGemmaLearnerConfig())
    if not isinstance(value, Mapping) or set(value) != set(defaults):
        raise SecFilingGemmaLearnerError("Learner config state is invalid")
    decoded: dict[str, Any] = {}
    for name, expected in defaults.items():
        observed = value[name]
        if isinstance(expected, int):
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed != expected
            ):
                raise SecFilingGemmaLearnerError("Learner config state is invalid")
            decoded[name] = observed
        else:
            decoded[name] = _decode_float_hex(observed, f"config.{name}")
    try:
        config = SecFilingGemmaLearnerConfig(**decoded)
    except TypeError as exc:
        raise SecFilingGemmaLearnerError("Learner config state is invalid") from exc
    config.validate()
    return config


def _encode_vector(values: np.ndarray) -> list[str]:
    return [_float_hex(item) for item in np.asarray(values, dtype=float)]


def _decode_vector(value: Any, location: str, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise SecFilingGemmaLearnerError(
            f"{location} must contain exactly {length} values"
        )
    return np.asarray(
        [_decode_float_hex(item, f"{location}[{index}]") for index, item in enumerate(value)],
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


def _design(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(matrix), dtype=float), matrix))


def _penalty(dimension: int, ridge_lambda: float) -> np.ndarray:
    diagonal = np.full(dimension, ridge_lambda, dtype=float)
    diagonal[0] = 0.0
    return diagonal


def _logistic_objective(
    design: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
    ridge_lambda: float,
) -> float:
    linear = design @ coefficients
    loss = float(np.mean(np.logaddexp(0.0, linear) - target * linear))
    return loss + 0.5 * ridge_lambda * float(
        coefficients[1:] @ coefficients[1:]
    )


def _fit_logistic(
    design: np.ndarray,
    target: np.ndarray,
    config: SecFilingGemmaLearnerConfig,
) -> tuple[np.ndarray, int]:
    prevalence = float(target.mean())
    coefficients = np.zeros(design.shape[1], dtype=float)
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = _penalty(design.shape[1], config.ridge_lambda)
    penalty_matrix = np.diag(penalty)
    for iteration in range(1, config.logistic_max_iterations + 1):
        probability = _stable_sigmoid(design @ coefficients)
        gradient = design.T @ (probability - target) / len(target)
        gradient += penalty * coefficients
        if float(np.max(np.abs(gradient))) <= config.logistic_tolerance:
            return coefficients, iteration - 1
        curvature = np.maximum(probability * (1.0 - probability), 1e-15)
        hessian = design.T @ (design * curvature[:, None]) / len(target)
        hessian += penalty_matrix
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        objective = _logistic_objective(
            design, target, coefficients, config.ridge_lambda
        )
        directional = float(gradient @ direction)
        step = 1.0
        accepted = False
        candidate = coefficients
        for _ in range(config.newton_line_search_max_steps):
            candidate = coefficients - step * direction
            if _logistic_objective(
                design, target, candidate, config.ridge_lambda
            ) <= objective - 1e-4 * step * directional:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise SecFilingGemmaLearnerError(
                "Ridge-logistic line search did not converge"
            )
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        if change <= config.logistic_tolerance:
            return coefficients, iteration
    raise SecFilingGemmaLearnerError(
        "Ridge-logistic solver exceeded its frozen iteration cap"
    )


def _fit_huber(
    design: np.ndarray,
    target: np.ndarray,
    config: SecFilingGemmaLearnerConfig,
) -> tuple[np.ndarray, int]:
    coefficients = np.zeros(design.shape[1], dtype=float)
    coefficients[0] = float(np.median(target))
    penalty = np.diag(_penalty(design.shape[1], config.ridge_lambda))
    for iteration in range(1, config.huber_max_iterations + 1):
        residual = target - design @ coefficients
        absolute = np.abs(residual)
        weights = np.ones(len(residual), dtype=float)
        outside = absolute > config.huber_delta
        weights[outside] = config.huber_delta / absolute[outside]
        system = design.T @ (design * weights[:, None]) / len(design)
        system += penalty
        right = design.T @ (weights * target) / len(design)
        try:
            candidate = np.linalg.solve(system, right)
        except np.linalg.LinAlgError:
            candidate = np.linalg.lstsq(system, right, rcond=None)[0]
        if not np.isfinite(candidate).all():
            raise SecFilingGemmaLearnerError(
                "Ridge-Huber solver produced non-finite coefficients"
            )
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        if change <= config.huber_tolerance:
            return coefficients, iteration
    raise SecFilingGemmaLearnerError(
        "Ridge-Huber solver exceeded its frozen iteration cap"
    )


class SecFilingGemmaTwoHeadLearner:
    """One deterministic linear probability/edge model with immutable state."""

    def __init__(self, config: SecFilingGemmaLearnerConfig | None = None) -> None:
        self.config = config or SecFilingGemmaLearnerConfig()
        self.config.validate()
        self.feature_names: tuple[str, ...] = ()
        self.raw_medians = np.empty(0, dtype=float)
        self.raw_scales = np.empty(0, dtype=float)
        self.target_center = 0.0
        self.target_scale = 1.0
        self.logistic_coefficients = np.empty(0, dtype=float)
        self.huber_coefficients = np.empty(0, dtype=float)
        self.logistic_iterations = 0
        self.huber_iterations = 0
        self.training_positive_count = 0
        self.fit_metadata: SecFilingGemmaFitMetadata | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            self.fit_metadata is not None
            and len(self.feature_names) > 0
            and len(self.logistic_coefficients) == len(self.feature_names) + 1
            and len(self.huber_coefficients) == len(self.feature_names) + 1
        )

    @property
    def model_sha256(self) -> str:
        return self.to_state()["state_sha256"]

    @staticmethod
    def _coerce_matrix(
        features: Any,
        feature_names: Sequence[str] | None,
        *,
        minimum_rows: int,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        inferred: tuple[str, ...] | None = None
        if hasattr(features, "columns") and hasattr(features, "to_numpy"):
            inferred = tuple(str(item) for item in features.columns)
            matrix = np.asarray(features.to_numpy(dtype=float), dtype=float)
        else:
            matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1 and minimum_rows == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or len(matrix) < minimum_rows or matrix.shape[1] < 1:
            raise SecFilingGemmaLearnerError("Feature matrix has invalid shape")
        names = inferred if feature_names is None else tuple(feature_names)
        if names is None or len(names) != matrix.shape[1]:
            raise SecFilingGemmaLearnerError("Feature schema does not match the matrix")
        if inferred is not None and names != inferred:
            raise SecFilingGemmaLearnerError("Feature columns were reordered")
        if any(not isinstance(name, str) or not name for name in names):
            raise SecFilingGemmaLearnerError("Feature names must be nonblank strings")
        if len(set(names)) != len(names):
            raise SecFilingGemmaLearnerError("Feature names must be unique")
        if not np.isfinite(matrix).all():
            raise SecFilingGemmaLearnerError("Features must all be finite")
        return matrix, names

    def fit(
        self,
        features: Any,
        binary_cash_win_target: Sequence[int | float],
        active_log_edge_10bps: Sequence[int | float],
        *,
        feature_names: Sequence[str] | None,
        fit_metadata: Mapping[str, Any] | SecFilingGemmaFitMetadata,
    ) -> "SecFilingGemmaTwoHeadLearner":
        self.config.validate()
        matrix, names = self._coerce_matrix(
            features, feature_names, minimum_rows=2
        )
        binary = np.asarray(binary_cash_win_target, dtype=float)
        edge = np.asarray(active_log_edge_10bps, dtype=float)
        if binary.shape != (len(matrix),) or edge.shape != (len(matrix),):
            raise SecFilingGemmaLearnerError(
                "Each training target must have one value per feature row"
            )
        if not np.isfinite(binary).all() or not np.isfinite(edge).all():
            raise SecFilingGemmaLearnerError("Training targets must be finite")
        if not np.isin(binary, (0.0, 1.0)).all():
            raise SecFilingGemmaLearnerError("Binary cash-win target must be zero or one")
        positive_count = int(binary.sum())
        if positive_count in (0, len(binary)):
            raise SecFilingGemmaLearnerError(
                "Ridge-logistic training requires both target classes"
            )
        metadata = SecFilingGemmaFitMetadata.coerce(fit_metadata)
        if metadata.training_row_count != len(matrix):
            raise SecFilingGemmaLearnerError(
                "Fit metadata row count does not match the training matrix"
            )
        if metadata.feature_schema_sha256 != canonical_sha256(list(names)):
            raise SecFilingGemmaLearnerError(
                "Fit metadata does not bind the exact ordered feature schema"
            )

        medians = np.median(matrix, axis=0)
        mads = np.median(np.abs(matrix - medians), axis=0)
        scales = np.maximum(
            self.config.raw_mad_multiplier * mads,
            self.config.raw_scale_floor,
        )
        scaled = np.clip(
            (matrix - medians) / scales,
            -self.config.raw_z_clip,
            self.config.raw_z_clip,
        )
        design = _design(scaled)
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
        logistic, logistic_iterations = _fit_logistic(
            design, binary, self.config
        )
        huber, huber_iterations = _fit_huber(
            design, standardized_edge, self.config
        )

        self.feature_names = names
        self.raw_medians = medians.astype(float, copy=True)
        self.raw_scales = scales.astype(float, copy=True)
        self.target_center = target_center
        self.target_scale = target_scale
        self.logistic_coefficients = logistic.astype(float, copy=True)
        self.huber_coefficients = huber.astype(float, copy=True)
        self.logistic_iterations = logistic_iterations
        self.huber_iterations = huber_iterations
        self.training_positive_count = positive_count
        self.fit_metadata = metadata
        self._validate_fitted()
        return self

    def _validate_fitted(self) -> None:
        if not self.is_fitted:
            raise SecFilingGemmaLearnerError("Learner is not fitted")
        assert self.fit_metadata is not None
        self.fit_metadata.validate()
        self.config.validate()
        arrays = (
            self.raw_medians,
            self.raw_scales,
            self.logistic_coefficients,
            self.huber_coefficients,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise SecFilingGemmaLearnerError("Learner state contains non-finite values")
        if np.any(self.raw_scales < self.config.raw_scale_floor):
            raise SecFilingGemmaLearnerError("Learner feature scale is below its floor")
        if (
            not math.isfinite(self.target_center)
            or not self.config.edge_clip_lower
            <= self.target_center
            <= self.config.edge_clip_upper
            or not math.isfinite(self.target_scale)
            or self.target_scale < self.config.target_scale_floor
        ):
            raise SecFilingGemmaLearnerError("Learner target scaling is invalid")
        if not (
            0 <= self.logistic_iterations <= self.config.logistic_max_iterations
            and 1 <= self.huber_iterations <= self.config.huber_max_iterations
        ):
            raise SecFilingGemmaLearnerError("Learner convergence counts are invalid")
        if not 0 < self.training_positive_count < self.fit_metadata.training_row_count:
            raise SecFilingGemmaLearnerError("Learner class counts are invalid")

    def _prediction_matrix(self, features: Any) -> np.ndarray:
        self._validate_fitted()
        matrix, names = self._coerce_matrix(
            features,
            self.feature_names,
            minimum_rows=1,
        )
        if names != self.feature_names:
            raise SecFilingGemmaLearnerError("Prediction feature schema changed")
        return matrix

    def predict_components(self, features: Any) -> dict[str, np.ndarray]:
        matrix = self._prediction_matrix(features)
        scaled = np.clip(
            (matrix - self.raw_medians) / self.raw_scales,
            -self.config.raw_z_clip,
            self.config.raw_z_clip,
        )
        design = _design(scaled)
        probability = _stable_sigmoid(design @ self.logistic_coefficients)
        standardized_edge = design @ self.huber_coefficients
        edge = self.target_center + self.target_scale * standardized_edge
        edge = np.clip(edge, self.config.edge_clip_lower, self.config.edge_clip_upper)
        return {
            "cash_win_probability_10bps": probability,
            "expected_active_log_edge_10bps": edge,
        }

    def _state_body(self) -> dict[str, Any]:
        self._validate_fitted()
        assert self.fit_metadata is not None
        parameters = {
            "raw_medians": _encode_vector(self.raw_medians),
            "raw_scales": _encode_vector(self.raw_scales),
            "target_center": _float_hex(self.target_center),
            "target_scale": _float_hex(self.target_scale),
            "logistic_coefficients": _encode_vector(self.logistic_coefficients),
            "huber_coefficients": _encode_vector(self.huber_coefficients),
        }
        return {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "config": _config_state(self.config),
            "feature_names": list(self.feature_names),
            "fit_metadata": asdict(self.fit_metadata),
            "training_positive_count": self.training_positive_count,
            "logistic_iterations": self.logistic_iterations,
            "huber_iterations": self.huber_iterations,
            "parameters": parameters,
            "parameters_sha256": canonical_sha256(parameters),
        }

    def to_state(self) -> dict[str, Any]:
        body = self._state_body()
        return {**body, "state_sha256": canonical_sha256(body)}

    def canonical_json(self) -> str:
        """Return the one canonical UTF-8 JSON representation for sealing."""

        return json.dumps(
            self.to_state(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "SecFilingGemmaTwoHeadLearner":
        if not isinstance(state, Mapping):
            raise SecFilingGemmaLearnerError("Learner state must be a mapping")
        expected_keys = {
            "state_schema_version",
            "model_type",
            "config",
            "feature_names",
            "fit_metadata",
            "training_positive_count",
            "logistic_iterations",
            "huber_iterations",
            "parameters",
            "parameters_sha256",
            "state_sha256",
        }
        if set(state) != expected_keys:
            raise SecFilingGemmaLearnerError("Learner state has invalid keys")
        body = {key: state[key] for key in expected_keys if key != "state_sha256"}
        observed_state_hash = _sha256(state["state_sha256"], "state_sha256")
        if not hmac.compare_digest(canonical_sha256(body), observed_state_hash):
            raise SecFilingGemmaLearnerError("Learner state checksum mismatch")
        if state["state_schema_version"] != STATE_SCHEMA_VERSION or state[
            "model_type"
        ] != MODEL_TYPE:
            raise SecFilingGemmaLearnerError("Learner state type or version changed")
        config = _config_from_state(state["config"])
        names = state["feature_names"]
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise SecFilingGemmaLearnerError("Learner feature schema is invalid")
        metadata = SecFilingGemmaFitMetadata.coerce(state["fit_metadata"])
        if metadata.feature_schema_sha256 != canonical_sha256(names):
            raise SecFilingGemmaLearnerError("Learner feature schema hash changed")
        parameters = state["parameters"]
        if not isinstance(parameters, Mapping) or set(parameters) != {
            "raw_medians",
            "raw_scales",
            "target_center",
            "target_scale",
            "logistic_coefficients",
            "huber_coefficients",
        }:
            raise SecFilingGemmaLearnerError("Learner parameters have invalid keys")
        if not hmac.compare_digest(
            canonical_sha256(parameters),
            _sha256(state["parameters_sha256"], "parameters_sha256"),
        ):
            raise SecFilingGemmaLearnerError("Learner parameter checksum mismatch")
        dimension = len(names)
        model = cls(config)
        model.feature_names = tuple(names)
        model.raw_medians = _decode_vector(parameters["raw_medians"], "raw_medians", dimension)
        model.raw_scales = _decode_vector(parameters["raw_scales"], "raw_scales", dimension)
        model.target_center = _decode_float_hex(parameters["target_center"], "target_center")
        model.target_scale = _decode_float_hex(parameters["target_scale"], "target_scale")
        model.logistic_coefficients = _decode_vector(
            parameters["logistic_coefficients"], "logistic_coefficients", dimension + 1
        )
        model.huber_coefficients = _decode_vector(
            parameters["huber_coefficients"], "huber_coefficients", dimension + 1
        )
        for name in (
            "training_positive_count",
            "logistic_iterations",
            "huber_iterations",
        ):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SecFilingGemmaLearnerError(f"{name} must be a nonnegative integer")
            setattr(model, name, value)
        model.fit_metadata = metadata
        model._validate_fitted()
        if model.to_state() != dict(state):
            raise SecFilingGemmaLearnerError("Learner state is not canonical")
        return model


def beta_1_1_climatology(positive_count: int, sample_count: int) -> float:
    if (
        isinstance(positive_count, bool)
        or isinstance(sample_count, bool)
        or not isinstance(positive_count, int)
        or not isinstance(sample_count, int)
        or sample_count < 0
        or positive_count < 0
        or positive_count > sample_count
    ):
        raise SecFilingGemmaLearnerError("Invalid climatology counts")
    return (positive_count + 1.0) / (sample_count + 2.0)


__all__ = [
    "HEAD_VARIANTS",
    "MODEL_TYPE",
    "STATE_SCHEMA_VERSION",
    "SecFilingGemmaFitMetadata",
    "SecFilingGemmaLearnerConfig",
    "SecFilingGemmaLearnerError",
    "SecFilingGemmaTwoHeadLearner",
    "beta_1_1_climatology",
]
