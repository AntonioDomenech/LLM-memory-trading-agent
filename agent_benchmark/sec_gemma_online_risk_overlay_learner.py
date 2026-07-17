"""Causal adapter around the exact pinned SEC/Gemma NumPy learner.

The adapter has no filesystem, network, model-runtime, market, clock, or
outcome I/O.  Its only outcome-bearing inputs are externally pinned lesson
rows whose counterfactual results have already matured.  Every persisted
floating-point value is a canonical ``float.hex`` string.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
from datetime import date
import hmac
import json
import math
from numbers import Real
import re
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    EXPECTED_EDGE_GATE,
    FEATURES,
    MARKET_FEATURES,
    MEANING_FEATURES,
    MINIMUM_CLASS_ROWS,
    MINIMUM_TRAINING_ROWS,
    POSITIVE_EDGE_TOLERANCE,
    PROBABILITY_GATE,
    SOURCE_PINS,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_features import (
    FEATURE_ROW_SCHEMA_VERSION,
    MARKET_RESULT_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_learner import (
    SecFilingGemmaLearnerError as _PinnedSecFilingGemmaLearnerError,
    SecFilingGemmaTwoHeadLearner as _PinnedSecFilingGemmaTwoHeadLearner,
)


LEARNER_ADAPTER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-learner-v1"
)
LESSON_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-lesson-row-v1"
)
FIT_AUDIT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-fit-audit-v1"
)
PREDICTION_AUDIT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-prediction-audit-v1"
)
GATE_AUDIT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-gate-audit-v1"
)
FEATURE_ARMS: Final[tuple[str, ...]] = (
    "semantic",
    "no_filing_meaning",
    "no_gemma_channel",
)
ARM_VECTOR_FIELDS: Final[dict[str, str]] = {
    "semantic": "semantic_values_hex",
    "no_filing_meaning": "no_filing_meaning_values_hex",
    "no_gemma_channel": "no_gemma_channel_values_hex",
}
ARM_HEAD_VARIANTS: Final[dict[str, str]] = {
    "semantic": "semantic",
    "no_filing_meaning": "ablation",
    "no_gemma_channel": "ablation",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FEATURE_ROW_KEYS = {
    "schema_version",
    "contract_version",
    "contract_sha256",
    "accession_number",
    "form",
    "decision_session",
    "acceptance_datetime",
    "artifact_stage",
    "extraction_status",
    "extraction_authenticated",
    "schema_valid_extraction",
    "document_quality",
    "market_available",
    "prediction_available",
    "fit_eligible",
    "meaning_nonzero",
    "unavailable_reasons",
    "unavailable_reasons_sha256",
    "market_result",
    "market_result_sha256",
    "meaning_feature_names",
    "meaning_values_hex",
    "semantic_quality_risk_hex",
    "form_10k_hex",
    "feature_names",
    "semantic_values_hex",
    "no_filing_meaning_values_hex",
    "no_gemma_channel_values_hex",
    "upstream_bindings",
    "upstream_bindings_sha256",
    "feature_row_sha256",
}
_LESSON_ROW_KEYS = {
    "schema_version",
    "contract_version",
    "contract_sha256",
    "accession_number",
    "decision_session",
    "maturity_session",
    "feature_row",
    "feature_row_sha256",
    "trainable",
    "binary_cash_win_target",
    "active_log_edge_10bps_hex",
    "label_evidence_sha256",
    "lesson_row_sha256",
}
_ADMITTED_LESSON_KEYS = {
    "lesson_row_sha256",
    "feature_row_sha256",
    "accession_number",
    "decision_session",
    "acceptance_datetime",
    "maturity_session",
    "trainable",
}
_TRAINING_MEMBER_KEYS = {
    *_ADMITTED_LESSON_KEYS,
    "binary_cash_win_target",
    "active_log_edge_10bps_hex",
    "feature_values_hex",
}
_FIT_AUDIT_KEYS = {
    "schema_version",
    "contract_version",
    "contract_sha256",
    "as_of_session",
    "arm",
    "candidate_sha256",
    "head_variant",
    "feature_names",
    "feature_schema_sha256",
    "learner_primitive_sha256",
    "supplied_matured_lesson_count",
    "audit_only_lesson_count",
    "training_row_count",
    "training_positive_count",
    "training_negative_count",
    "minimum_training_rows",
    "minimum_rows_per_binary_class",
    "learner_ready",
    "admitted_lessons",
    "admitted_lessons_sha256",
    "training_members",
    "training_membership_sha256",
    "fit_state",
    "fit_state_sha256",
    "fit_audit_sha256",
}
_PREDICTION_ROW_KEYS = {
    "schema_version",
    "contract_version",
    "contract_sha256",
    "decision_session",
    "accession_number",
    "acceptance_datetime",
    "feature_row_sha256",
    "arm",
    "fit_as_of_session",
    "frozen_fit_reuse",
    "learner_ready",
    "current_feature_available",
    "prediction_available",
    "fitted_prediction_available",
    "action_available",
    "raw_gate_pass",
    "gate_schedule_overlay",
    "gate_audit",
    "gate_audit_sha256",
    "fit_audit",
    "fit_audit_sha256",
    "prediction_row_sha256",
}


class SecFilingGemmaLearnerError(ValueError):
    """The frozen pure-stdlib learner primitive rejected its input."""


@dataclass(frozen=True)
class _LearnerConfig:
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
        expected = asdict(_LearnerConfig())
        observed = asdict(self)
        if set(observed) != set(expected):
            raise SecFilingGemmaLearnerError(
                "Learner configuration differs from the frozen design"
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
                    and isinstance(value, Real)
                    and math.isfinite(float(value))
                    and float(value).hex()
                    == float(expected_value).hex()
                )
            if not matches:
                raise SecFilingGemmaLearnerError(
                    "Learner configuration differs from the frozen design"
                )


@dataclass(frozen=True)
class _FitMetadata:
    candidate_sha256: str
    head_variant: str
    fold_id: str
    train_label_maturity_through: str
    training_set_sha256: str
    training_row_count: int
    maximum_training_label_maturity_session: str
    feature_schema_sha256: str

    @classmethod
    def coerce(cls, value: Any) -> "_FitMetadata":
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            if set(value) != set(cls.__dataclass_fields__):
                raise SecFilingGemmaLearnerError(
                    "Fit metadata has invalid keys"
                )
            try:
                result = cls(**dict(value))
            except TypeError as exc:
                raise SecFilingGemmaLearnerError(
                    "Fit metadata is invalid"
                ) from exc
        else:
            raise SecFilingGemmaLearnerError(
                "Fit metadata must be a mapping"
            )
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
            "candidate_sha256",
            "training_set_sha256",
            "feature_schema_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise SecFilingGemmaLearnerError(
                    f"{name} must be a lowercase SHA-256 digest"
                )
        if self.head_variant not in {"semantic", "ablation"}:
            raise SecFilingGemmaLearnerError(
                "Unknown learner head variant"
            )
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise SecFilingGemmaLearnerError(
                "fold_id must be nonblank"
            )
        try:
            cutoff = date.fromisoformat(
                self.train_label_maturity_through
            )
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


def _primitive_float(value: Any, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise SecFilingGemmaLearnerError(
            f"{location} must be a finite float"
        )
    return float(value)


def _primitive_float_hex(value: Any, location: str = "float") -> str:
    return _primitive_float(value, location).hex()


def _primitive_decode_float_hex(value: Any, location: str) -> float:
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


def _primitive_config_state(config: _LearnerConfig) -> dict[str, Any]:
    config.validate()
    return {
        name: (
            value
            if isinstance(value, int)
            else _primitive_float_hex(value, f"config.{name}")
        )
        for name, value in asdict(config).items()
    }


def _primitive_config_from_state(value: Any) -> _LearnerConfig:
    defaults = asdict(_LearnerConfig())
    if not isinstance(value, Mapping) or set(value) != set(defaults):
        raise SecFilingGemmaLearnerError(
            "Learner config state is invalid"
        )
    decoded: dict[str, Any] = {}
    for name, expected in defaults.items():
        observed = value[name]
        if isinstance(expected, int):
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed != expected
            ):
                raise SecFilingGemmaLearnerError(
                    "Learner config state is invalid"
                )
            decoded[name] = observed
        else:
            decoded[name] = _primitive_decode_float_hex(
                observed, f"config.{name}"
            )
    config = _LearnerConfig(**decoded)
    config.validate()
    return config


def _median(values: Sequence[float]) -> float:
    if not values:
        raise SecFilingGemmaLearnerError(
            "Median requires at least one value"
        )
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        first * second
        for first, second in zip(left, right, strict=True)
    )


def _matvec(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _solve_linear(
    matrix: Sequence[Sequence[float]],
    right: Sequence[float],
) -> list[float]:
    dimension = len(right)
    if (
        len(matrix) != dimension
        or any(len(row) != dimension for row in matrix)
    ):
        raise SecFilingGemmaLearnerError(
            "Learner linear system has invalid shape"
        )
    augmented = [
        [float(value) for value in row] + [float(right[index])]
        for index, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot = max(
            range(column, dimension),
            key=lambda row: abs(augmented[row][column]),
        )
        pivot_value = augmented[pivot][column]
        if not math.isfinite(pivot_value) or abs(pivot_value) <= 1e-18:
            raise SecFilingGemmaLearnerError(
                "Learner linear system is singular"
            )
        if pivot != column:
            augmented[column], augmented[pivot] = (
                augmented[pivot],
                augmented[column],
            )
        for row in range(column + 1, dimension):
            factor = augmented[row][column] / augmented[column][column]
            augmented[row][column] = 0.0
            for offset in range(column + 1, dimension + 1):
                augmented[row][offset] -= (
                    factor * augmented[column][offset]
                )
    result = [0.0] * dimension
    for row in range(dimension - 1, -1, -1):
        numerator = augmented[row][dimension] - sum(
            augmented[row][column] * result[column]
            for column in range(row + 1, dimension)
        )
        result[row] = numerator / augmented[row][row]
    if any(not math.isfinite(value) for value in result):
        raise SecFilingGemmaLearnerError(
            "Learner linear system produced non-finite coefficients"
        )
    return result


def _design(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[1.0, *row] for row in matrix]


def _logistic_objective(
    design: Sequence[Sequence[float]],
    target: Sequence[float],
    coefficients: Sequence[float],
    ridge_lambda: float,
) -> float:
    losses = []
    for row, expected in zip(design, target, strict=True):
        linear = _dot(row, coefficients)
        losses.append(
            max(0.0, linear)
            + math.log1p(math.exp(-abs(linear)))
            - expected * linear
        )
    return sum(losses) / len(losses) + 0.5 * ridge_lambda * sum(
        value * value for value in coefficients[1:]
    )


def _fit_logistic(
    design: Sequence[Sequence[float]],
    target: Sequence[float],
    config: _LearnerConfig,
) -> tuple[list[float], int]:
    prevalence = sum(target) / len(target)
    coefficients = [0.0] * len(design[0])
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalties = [0.0] + [
        config.ridge_lambda
    ] * (len(coefficients) - 1)
    for iteration in range(1, config.logistic_max_iterations + 1):
        probability = [
            _stable_sigmoid(value)
            for value in _matvec(design, coefficients)
        ]
        gradient = []
        for column in range(len(coefficients)):
            value = sum(
                row[column] * (probability[index] - target[index])
                for index, row in enumerate(design)
            ) / len(target)
            gradient.append(
                value + penalties[column] * coefficients[column]
            )
        if max(abs(value) for value in gradient) <= (
            config.logistic_tolerance
        ):
            return coefficients, iteration - 1
        curvature = [
            max(value * (1.0 - value), 1e-15)
            for value in probability
        ]
        hessian: list[list[float]] = []
        for first in range(len(coefficients)):
            row_values: list[float] = []
            for second in range(len(coefficients)):
                value = sum(
                    row[first] * row[second] * curvature[index]
                    for index, row in enumerate(design)
                ) / len(target)
                if first == second:
                    value += penalties[first]
                row_values.append(value)
            hessian.append(row_values)
        direction = _solve_linear(hessian, gradient)
        objective = _logistic_objective(
            design, target, coefficients, config.ridge_lambda
        )
        directional = _dot(gradient, direction)
        step = 1.0
        accepted = False
        candidate = coefficients
        for _ in range(config.newton_line_search_max_steps):
            candidate = [
                value - step * delta
                for value, delta in zip(
                    coefficients, direction, strict=True
                )
            ]
            if _logistic_objective(
                design,
                target,
                candidate,
                config.ridge_lambda,
            ) <= objective - 1e-4 * step * directional:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise SecFilingGemmaLearnerError(
                "Ridge-logistic line search did not converge"
            )
        change = max(
            abs(new - old)
            for new, old in zip(candidate, coefficients, strict=True)
        )
        coefficients = candidate
        if change <= config.logistic_tolerance:
            return coefficients, iteration
    raise SecFilingGemmaLearnerError(
        "Ridge-logistic solver exceeded its frozen iteration cap"
    )


def _fit_huber(
    design: Sequence[Sequence[float]],
    target: Sequence[float],
    config: _LearnerConfig,
) -> tuple[list[float], int]:
    coefficients = [0.0] * len(design[0])
    coefficients[0] = _median(target)
    penalties = [0.0] + [
        config.ridge_lambda
    ] * (len(coefficients) - 1)
    for iteration in range(1, config.huber_max_iterations + 1):
        predicted = _matvec(design, coefficients)
        residual = [
            actual - estimate
            for actual, estimate in zip(target, predicted, strict=True)
        ]
        weights = [
            (
                1.0
                if abs(value) <= config.huber_delta
                else config.huber_delta / abs(value)
            )
            for value in residual
        ]
        system: list[list[float]] = []
        for first in range(len(coefficients)):
            row_values: list[float] = []
            for second in range(len(coefficients)):
                value = sum(
                    row[first] * row[second] * weights[index]
                    for index, row in enumerate(design)
                ) / len(design)
                if first == second:
                    value += penalties[first]
                row_values.append(value)
            system.append(row_values)
        right = [
            sum(
                row[column] * weights[index] * target[index]
                for index, row in enumerate(design)
            )
            / len(design)
            for column in range(len(coefficients))
        ]
        candidate = _solve_linear(system, right)
        change = max(
            abs(new - old)
            for new, old in zip(candidate, coefficients, strict=True)
        )
        coefficients = candidate
        if change <= config.huber_tolerance:
            return coefficients, iteration
    raise SecFilingGemmaLearnerError(
        "Ridge-Huber solver exceeded its frozen iteration cap"
    )


class SecFilingGemmaTwoHeadLearner:
    """Pure-stdlib numerical equivalent of the pinned two-head learner."""

    _MODEL_TYPE: Final[str] = "aapl_sec_filing_gemma_linear_two_head_v1"
    _STATE_SCHEMA_VERSION: Final[int] = 1

    def __init__(self, config: _LearnerConfig | None = None) -> None:
        self.config = config or _LearnerConfig()
        self.config.validate()
        self.feature_names: tuple[str, ...] = ()
        self.raw_medians: list[float] = []
        self.raw_scales: list[float] = []
        self.target_center = 0.0
        self.target_scale = 1.0
        self.logistic_coefficients: list[float] = []
        self.huber_coefficients: list[float] = []
        self.logistic_iterations = 0
        self.huber_iterations = 0
        self.training_positive_count = 0
        self.fit_metadata: _FitMetadata | None = None

    @property
    def is_fitted(self) -> bool:
        return (
            self.fit_metadata is not None
            and bool(self.feature_names)
            and len(self.logistic_coefficients)
            == len(self.feature_names) + 1
            and len(self.huber_coefficients)
            == len(self.feature_names) + 1
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
    ) -> tuple[list[list[float]], tuple[str, ...]]:
        inferred: tuple[str, ...] | None = None
        if hasattr(features, "columns") and hasattr(features, "to_numpy"):
            inferred = tuple(str(item) for item in features.columns)
            raw = features.to_numpy(dtype=float).tolist()
        else:
            raw = list(features)
            if minimum_rows == 1 and raw and isinstance(raw[0], Real):
                raw = [raw]
        if len(raw) < minimum_rows:
            raise SecFilingGemmaLearnerError(
                "Feature matrix has invalid shape"
            )
        matrix: list[list[float]] = []
        width: int | None = None
        for row in raw:
            if isinstance(row, (str, bytes)) or not isinstance(
                row, Sequence
            ):
                raise SecFilingGemmaLearnerError(
                    "Feature matrix has invalid shape"
                )
            values = [
                _primitive_float(value, "feature value") for value in row
            ]
            if width is None:
                width = len(values)
            if not values or len(values) != width:
                raise SecFilingGemmaLearnerError(
                    "Feature matrix has invalid shape"
                )
            matrix.append(values)
        names = inferred if feature_names is None else tuple(feature_names)
        if names is None or width is None or len(names) != width:
            raise SecFilingGemmaLearnerError(
                "Feature schema does not match the matrix"
            )
        if inferred is not None and names != inferred:
            raise SecFilingGemmaLearnerError(
                "Feature columns were reordered"
            )
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise SecFilingGemmaLearnerError(
                "Feature names must be unique nonblank strings"
            )
        return matrix, names

    def fit(
        self,
        features: Any,
        binary_cash_win_target: Sequence[int | float],
        active_log_edge_10bps: Sequence[int | float],
        *,
        feature_names: Sequence[str] | None,
        fit_metadata: Mapping[str, Any] | _FitMetadata,
    ) -> "SecFilingGemmaTwoHeadLearner":
        self.config.validate()
        matrix, names = self._coerce_matrix(
            features,
            feature_names,
            minimum_rows=2,
        )
        binary = [
            _primitive_float(value, "binary target")
            for value in binary_cash_win_target
        ]
        edge = [
            _primitive_float(value, "edge target")
            for value in active_log_edge_10bps
        ]
        if len(binary) != len(matrix) or len(edge) != len(matrix):
            raise SecFilingGemmaLearnerError(
                "Each training target must have one value per feature row"
            )
        if any(value not in (0.0, 1.0) for value in binary):
            raise SecFilingGemmaLearnerError(
                "Binary cash-win target must be zero or one"
            )
        positive_count = int(sum(binary))
        if positive_count in (0, len(binary)):
            raise SecFilingGemmaLearnerError(
                "Ridge-logistic training requires both target classes"
            )
        metadata = _FitMetadata.coerce(fit_metadata)
        if metadata.training_row_count != len(matrix):
            raise SecFilingGemmaLearnerError(
                "Fit metadata row count does not match the training matrix"
            )
        if metadata.feature_schema_sha256 != canonical_sha256(list(names)):
            raise SecFilingGemmaLearnerError(
                "Fit metadata does not bind the exact ordered feature schema"
            )
        medians = [
            _median([row[column] for row in matrix])
            for column in range(len(names))
        ]
        mads = [
            _median(
                [
                    abs(row[column] - medians[column])
                    for row in matrix
                ]
            )
            for column in range(len(names))
        ]
        scales = [
            max(
                self.config.raw_mad_multiplier * mad,
                self.config.raw_scale_floor,
            )
            for mad in mads
        ]
        scaled = [
            [
                max(
                    -self.config.raw_z_clip,
                    min(
                        self.config.raw_z_clip,
                        (row[column] - medians[column])
                        / scales[column],
                    ),
                )
                for column in range(len(names))
            ]
            for row in matrix
        ]
        design = _design(scaled)
        clipped_edge = [
            max(
                self.config.edge_clip_lower,
                min(self.config.edge_clip_upper, value),
            )
            for value in edge
        ]
        target_center = _median(clipped_edge)
        target_mad = _median(
            [abs(value - target_center) for value in clipped_edge]
        )
        target_scale = max(
            self.config.target_mad_multiplier * target_mad,
            self.config.target_scale_floor,
        )
        standardized_edge = [
            (value - target_center) / target_scale
            for value in clipped_edge
        ]
        logistic, logistic_iterations = _fit_logistic(
            design, binary, self.config
        )
        huber, huber_iterations = _fit_huber(
            design, standardized_edge, self.config
        )
        self.feature_names = names
        self.raw_medians = medians
        self.raw_scales = scales
        self.target_center = target_center
        self.target_scale = target_scale
        self.logistic_coefficients = logistic
        self.huber_coefficients = huber
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
        if any(
            not math.isfinite(value)
            for array in arrays
            for value in array
        ):
            raise SecFilingGemmaLearnerError(
                "Learner state contains non-finite values"
            )
        if any(
            value < self.config.raw_scale_floor
            for value in self.raw_scales
        ):
            raise SecFilingGemmaLearnerError(
                "Learner feature scale is below its floor"
            )
        if (
            not math.isfinite(self.target_center)
            or not self.config.edge_clip_lower
            <= self.target_center
            <= self.config.edge_clip_upper
            or not math.isfinite(self.target_scale)
            or self.target_scale < self.config.target_scale_floor
        ):
            raise SecFilingGemmaLearnerError(
                "Learner target scaling is invalid"
            )
        if not (
            0
            <= self.logistic_iterations
            <= self.config.logistic_max_iterations
            and 1
            <= self.huber_iterations
            <= self.config.huber_max_iterations
        ):
            raise SecFilingGemmaLearnerError(
                "Learner convergence counts are invalid"
            )
        if not (
            0
            < self.training_positive_count
            < self.fit_metadata.training_row_count
        ):
            raise SecFilingGemmaLearnerError(
                "Learner class counts are invalid"
            )

    def predict_components(
        self,
        features: Any,
    ) -> dict[str, list[float]]:
        self._validate_fitted()
        matrix, names = self._coerce_matrix(
            features,
            self.feature_names,
            minimum_rows=1,
        )
        if names != self.feature_names:
            raise SecFilingGemmaLearnerError(
                "Prediction feature schema changed"
            )
        scaled = [
            [
                max(
                    -self.config.raw_z_clip,
                    min(
                        self.config.raw_z_clip,
                        (row[column] - self.raw_medians[column])
                        / self.raw_scales[column],
                    ),
                )
                for column in range(len(self.feature_names))
            ]
            for row in matrix
        ]
        design = _design(scaled)
        probability = [
            _stable_sigmoid(
                _dot(row, self.logistic_coefficients)
            )
            for row in design
        ]
        edge = [
            max(
                self.config.edge_clip_lower,
                min(
                    self.config.edge_clip_upper,
                    self.target_center
                    + self.target_scale
                    * _dot(row, self.huber_coefficients),
                ),
            )
            for row in design
        ]
        return {
            "cash_win_probability_10bps": probability,
            "expected_active_log_edge_10bps": edge,
        }

    def _state_body(self) -> dict[str, Any]:
        self._validate_fitted()
        assert self.fit_metadata is not None
        parameters = {
            "raw_medians": [
                _primitive_float_hex(value)
                for value in self.raw_medians
            ],
            "raw_scales": [
                _primitive_float_hex(value)
                for value in self.raw_scales
            ],
            "target_center": _primitive_float_hex(
                self.target_center
            ),
            "target_scale": _primitive_float_hex(self.target_scale),
            "logistic_coefficients": [
                _primitive_float_hex(value)
                for value in self.logistic_coefficients
            ],
            "huber_coefficients": [
                _primitive_float_hex(value)
                for value in self.huber_coefficients
            ],
        }
        return {
            "state_schema_version": self._STATE_SCHEMA_VERSION,
            "model_type": self._MODEL_TYPE,
            "config": _primitive_config_state(self.config),
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
        return json.dumps(
            self.to_state(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
    ) -> "SecFilingGemmaTwoHeadLearner":
        if not isinstance(state, Mapping):
            raise SecFilingGemmaLearnerError(
                "Learner state must be a mapping"
            )
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
            raise SecFilingGemmaLearnerError(
                "Learner state has invalid keys"
            )
        body = {
            key: state[key]
            for key in expected_keys
            if key != "state_sha256"
        }
        observed_state_hash = state["state_sha256"]
        if (
            not isinstance(observed_state_hash, str)
            or _SHA256_RE.fullmatch(observed_state_hash) is None
            or not hmac.compare_digest(
                canonical_sha256(body), observed_state_hash
            )
        ):
            raise SecFilingGemmaLearnerError(
                "Learner state checksum mismatch"
            )
        if (
            state["state_schema_version"] != cls._STATE_SCHEMA_VERSION
            or state["model_type"] != cls._MODEL_TYPE
        ):
            raise SecFilingGemmaLearnerError(
                "Learner state type or version changed"
            )
        config = _primitive_config_from_state(state["config"])
        names = state["feature_names"]
        if (
            not isinstance(names, list)
            or not names
            or any(
                not isinstance(name, str) or not name for name in names
            )
            or len(set(names)) != len(names)
        ):
            raise SecFilingGemmaLearnerError(
                "Learner feature schema is invalid"
            )
        metadata = _FitMetadata.coerce(state["fit_metadata"])
        if metadata.feature_schema_sha256 != canonical_sha256(names):
            raise SecFilingGemmaLearnerError(
                "Learner feature schema hash changed"
            )
        parameters = state["parameters"]
        expected_parameters = {
            "raw_medians",
            "raw_scales",
            "target_center",
            "target_scale",
            "logistic_coefficients",
            "huber_coefficients",
        }
        if (
            not isinstance(parameters, Mapping)
            or set(parameters) != expected_parameters
        ):
            raise SecFilingGemmaLearnerError(
                "Learner parameters have invalid keys"
            )
        observed_parameters_hash = state["parameters_sha256"]
        if (
            not isinstance(observed_parameters_hash, str)
            or _SHA256_RE.fullmatch(observed_parameters_hash) is None
            or not hmac.compare_digest(
                canonical_sha256(parameters),
                observed_parameters_hash,
            )
        ):
            raise SecFilingGemmaLearnerError(
                "Learner parameter checksum mismatch"
            )
        dimension = len(names)

        def vector(value: Any, location: str, length: int) -> list[float]:
            if not isinstance(value, list) or len(value) != length:
                raise SecFilingGemmaLearnerError(
                    f"{location} must contain exactly {length} values"
                )
            return [
                _primitive_decode_float_hex(
                    item, f"{location}[{index}]"
                )
                for index, item in enumerate(value)
            ]

        model = cls(config)
        model.feature_names = tuple(names)
        model.raw_medians = vector(
            parameters["raw_medians"], "raw_medians", dimension
        )
        model.raw_scales = vector(
            parameters["raw_scales"], "raw_scales", dimension
        )
        model.target_center = _primitive_decode_float_hex(
            parameters["target_center"], "target_center"
        )
        model.target_scale = _primitive_decode_float_hex(
            parameters["target_scale"], "target_scale"
        )
        model.logistic_coefficients = vector(
            parameters["logistic_coefficients"],
            "logistic_coefficients",
            dimension + 1,
        )
        model.huber_coefficients = vector(
            parameters["huber_coefficients"],
            "huber_coefficients",
            dimension + 1,
        )
        for name in (
            "training_positive_count",
            "logistic_iterations",
            "huber_iterations",
        ):
            value = state[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise SecFilingGemmaLearnerError(
                    f"{name} must be a nonnegative integer"
                )
            setattr(model, name, value)
        model.fit_metadata = metadata
        model._validate_fitted()
        if model.to_state() != dict(state):
            raise SecFilingGemmaLearnerError(
                "Learner state is not canonical"
            )
        return model


# The overlay must execute the exact source-pinned NumPy primitive.  The
# historical pure-Python mirror above is deliberately unreachable from the
# module namespace so it cannot silently introduce last-bit solver drift.
SecFilingGemmaLearnerError = _PinnedSecFilingGemmaLearnerError
SecFilingGemmaTwoHeadLearner = _PinnedSecFilingGemmaTwoHeadLearner


class SecGemmaOnlineRiskOverlayLearnerError(ValueError):
    """Raised when causal learner inputs or audit state fail closed."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a mapping"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a canonical ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a canonical ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _canonical_float_hex(value: Any, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a finite number"
        )
    return number.hex()


def _decode_float_hex(value: Any, location: str) -> float:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a canonical hexadecimal float"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a canonical hexadecimal float"
        ) from exc
    if not math.isfinite(number) or number.hex() != value:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a canonical finite hexadecimal float"
        )
    return number


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be boolean"
        )
    return value


def _hash_body(
    value: Mapping[str, Any],
    hash_field: str,
    *,
    expected_sha256: str,
    location: str,
) -> str:
    observed = _sha256(value.get(hash_field), f"{location}.{hash_field}")
    expected = _sha256(expected_sha256, f"expected {location} hash")
    body = {key: value[key] for key in value if key != hash_field}
    calculated = canonical_sha256(body)
    if not hmac.compare_digest(observed, calculated):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} checksum mismatch"
        )
    if not hmac.compare_digest(observed, expected):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} is not the externally pinned immutable row"
        )
    return observed


def _validate_market_result(
    value: Any,
    *,
    expected_sha256: str,
) -> Mapping[str, Any]:
    result = _mapping(value, "feature row market result")
    expected_keys = {
        "schema_version",
        "available",
        "feature_names",
        "values_hex",
        "unavailable_reasons",
        "unavailable_reasons_sha256",
        "market_result_sha256",
    }
    if set(result) != expected_keys:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row market result keys changed"
        )
    if result["schema_version"] != MARKET_RESULT_SCHEMA_VERSION:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row market result schema changed"
        )
    available = _boolean(result["available"], "market result available")
    if result["feature_names"] != list(MARKET_FEATURES):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row market feature order changed"
        )
    reasons = result["unavailable_reasons"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(item, str) or not item for item in reasons)
        or reasons != sorted(set(reasons))
        or result["unavailable_reasons_sha256"] != canonical_sha256(reasons)
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row market unavailable reasons changed"
        )
    values = _mapping(result["values_hex"], "market values")
    if set(values) != set(MARKET_FEATURES):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row market values changed"
        )
    if available:
        if reasons:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Available market result contains unavailable reasons"
            )
        for name in MARKET_FEATURES:
            _decode_float_hex(values[name], f"market value {name}")
    elif not reasons or any(values[name] is not None for name in MARKET_FEATURES):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Unavailable market result exposes feature values"
        )
    _hash_body(
        result,
        "market_result_sha256",
        expected_sha256=expected_sha256,
        location="market result",
    )
    return result


def _decode_vector(
    value: Any,
    location: str,
    *,
    allow_none: bool,
) -> tuple[float, ...] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, list) or len(value) != len(FEATURES):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must contain the exact ordered feature vector"
        )
    return tuple(
        _decode_float_hex(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    )


def validate_online_overlay_feature_row(
    row: Mapping[str, Any],
    *,
    expected_feature_row_sha256: str,
) -> dict[str, Any]:
    """Validate a self-contained feature row against its external hash pin."""

    value = dict(_mapping(row, "feature row"))
    if set(value) != _FEATURE_ROW_KEYS:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row keys changed"
        )
    if (
        value["schema_version"] != FEATURE_ROW_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"] != CONTRACT_SHA256
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row experiment identity changed"
        )
    accession = value["accession_number"]
    if not isinstance(accession, str) or not accession:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row accession is invalid"
        )
    if value["form"] not in {"10-K", "10-Q"}:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row form is invalid"
        )
    _iso_date(value["decision_session"], "feature row decision session")
    acceptance = value["acceptance_datetime"]
    if acceptance is not None and (
        not isinstance(acceptance, str)
        or len(acceptance) != 14
        or not acceptance.isascii()
        or not acceptance.isdigit()
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row acceptance timestamp is invalid"
        )
    if not isinstance(value["artifact_stage"], str) or not value[
        "artifact_stage"
    ]:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row stage is invalid"
        )
    extraction_status = value["extraction_status"]
    if extraction_status not in {"valid", "invalid", "unavailable"}:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row extraction status is invalid"
        )
    extraction_authenticated = _boolean(
        value["extraction_authenticated"],
        "feature row extraction authenticated",
    )
    schema_valid = _boolean(
        value["schema_valid_extraction"],
        "feature row schema valid extraction",
    )
    market_available = _boolean(
        value["market_available"], "feature row market available"
    )
    prediction_available = _boolean(
        value["prediction_available"],
        "feature row prediction available",
    )
    fit_eligible = _boolean(
        value["fit_eligible"], "feature row fit eligible"
    )
    meaning_nonzero = _boolean(
        value["meaning_nonzero"], "feature row meaning nonzero"
    )
    if extraction_status == "valid":
        if (
            not extraction_authenticated
            or not schema_valid
            or value["document_quality"] not in {"usable", "thin", "unusable"}
        ):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Valid extraction feature state is inconsistent"
            )
    elif extraction_status == "invalid":
        if (
            not extraction_authenticated
            or schema_valid
            or value["document_quality"] is not None
        ):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Invalid extraction feature state is inconsistent"
            )
    elif (
        extraction_authenticated
        or schema_valid
        or value["document_quality"] is not None
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Unavailable extraction feature state is inconsistent"
        )
    expected_available = market_available and extraction_authenticated
    if (
        prediction_available != expected_available
        or fit_eligible != prediction_available
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row availability semantics changed"
        )
    reasons = value["unavailable_reasons"]
    expected_reasons: list[str] = []
    if not market_available:
        expected_reasons.append("market_unavailable")
    if not extraction_authenticated:
        expected_reasons.append(
            "missing_or_unauthenticated_extraction_evidence"
        )
    if (
        reasons != expected_reasons
        or value["unavailable_reasons_sha256"]
        != canonical_sha256(expected_reasons)
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row unavailable reasons changed"
        )
    market_hash = _sha256(
        value["market_result_sha256"], "feature row market result hash"
    )
    market_result = _validate_market_result(
        value["market_result"],
        expected_sha256=market_hash,
    )
    if bool(market_result["available"]) != market_available:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row market availability differs from its result"
        )
    if value["meaning_feature_names"] != list(MEANING_FEATURES):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row meaning feature order changed"
        )
    meaning_values = _mapping(
        value["meaning_values_hex"], "feature row meaning values"
    )
    if set(meaning_values) != set(MEANING_FEATURES):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row meaning values changed"
        )
    decoded_meaning = {
        name: _decode_float_hex(
            meaning_values[name], f"feature row meaning value {name}"
        )
        for name in MEANING_FEATURES
    }
    if any(
        number < -1.0 or number > 1.0
        for name, number in decoded_meaning.items()
        if name != "adverse_flag_fraction"
    ) or not 0.0 <= decoded_meaning["adverse_flag_fraction"] <= 1.0:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row meaning value is outside its frozen range"
        )
    quality = _decode_float_hex(
        value["semantic_quality_risk_hex"],
        "feature row semantic quality risk",
    )
    if quality not in {0.0, 0.5, 1.0}:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row semantic quality risk changed"
        )
    if extraction_status != "valid" and (
        any(decoded_meaning.values()) or quality != 1.0
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Unavailable or invalid extraction exposes filing meaning"
        )
    if extraction_status == "valid":
        expected_quality = {
            "usable": 0.0,
            "thin": 0.5,
            "unusable": 1.0,
        }[value["document_quality"]]
        if quality != expected_quality:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Feature row extraction quality encoding changed"
            )
        if value["document_quality"] == "unusable" and any(
            decoded_meaning.values()
        ):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Unusable extraction exposes filing meaning"
            )
    expected_nonzero = schema_valid and any(
        abs(number) > POSITIVE_EDGE_TOLERANCE
        for number in decoded_meaning.values()
    )
    if meaning_nonzero != expected_nonzero:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row nonzero-meaning flag changed"
        )
    form_10k = _decode_float_hex(
        value["form_10k_hex"], "feature row form control"
    )
    if form_10k != float(value["form"] == "10-K"):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row form control changed"
        )
    if value["feature_names"] != list(FEATURES):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row ordered schema changed"
        )
    semantic = _decode_vector(
        value["semantic_values_hex"],
        "semantic feature vector",
        allow_none=not prediction_available,
    )
    no_meaning = _decode_vector(
        value["no_filing_meaning_values_hex"],
        "no-filing-meaning feature vector",
        allow_none=not prediction_available,
    )
    no_gemma = _decode_vector(
        value["no_gemma_channel_values_hex"],
        "no-Gemma-channel feature vector",
        allow_none=not prediction_available,
    )
    if prediction_available:
        assert semantic is not None
        assert no_meaning is not None
        assert no_gemma is not None
        market_values = tuple(
            _decode_float_hex(
                market_result["values_hex"][name],
                f"market result value {name}",
            )
            for name in MARKET_FEATURES
        )
        expected_semantic = (
            *market_values,
            *(decoded_meaning[name] for name in MEANING_FEATURES),
            quality,
            form_10k,
        )
        expected_no_meaning = (
            *market_values,
            *(0.0 for _ in MEANING_FEATURES),
            quality,
            form_10k,
        )
        expected_no_gemma = (
            *market_values,
            *(0.0 for _ in MEANING_FEATURES),
            0.0,
            form_10k,
        )
        if (
            semantic != expected_semantic
            or no_meaning != expected_no_meaning
            or no_gemma != expected_no_gemma
        ):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Feature arm transform differs from preregistration"
            )
    elif any(
        item is not None for item in (semantic, no_meaning, no_gemma)
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Unavailable feature row exposes a prediction vector"
        )
    bindings = _mapping(
        value["upstream_bindings"], "feature row upstream bindings"
    )
    if value["upstream_bindings_sha256"] != canonical_sha256(bindings):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Feature row upstream binding checksum mismatch"
        )
    _hash_body(
        value,
        "feature_row_sha256",
        expected_sha256=expected_feature_row_sha256,
        location="feature row",
    )
    return copy.deepcopy(value)


def _arm(value: Any) -> str:
    if value not in FEATURE_ARMS:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Unknown SEC/Gemma online learner feature arm"
        )
    return str(value)


def _select_vector(
    feature_row: Mapping[str, Any],
    arm: str,
) -> tuple[float, ...] | None:
    selected = feature_row[ARM_VECTOR_FIELDS[arm]]
    return _decode_vector(
        selected,
        f"{arm} feature vector",
        allow_none=not bool(feature_row["prediction_available"]),
    )


def select_online_overlay_feature_arm(
    row: Mapping[str, Any],
    *,
    expected_feature_row_sha256: str,
    arm: str,
) -> dict[str, Any]:
    """Return one validated arm without changing any feature values."""

    selected_arm = _arm(arm)
    feature_row = validate_online_overlay_feature_row(
        row,
        expected_feature_row_sha256=expected_feature_row_sha256,
    )
    values = feature_row[ARM_VECTOR_FIELDS[selected_arm]]
    body = {
        "schema_version": LEARNER_ADAPTER_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "arm": selected_arm,
        "vector_field": ARM_VECTOR_FIELDS[selected_arm],
        "feature_row_sha256": feature_row["feature_row_sha256"],
        "feature_names": list(FEATURES),
        "values_hex": copy.deepcopy(values),
    }
    return {**body, "feature_arm_sha256": canonical_sha256(body)}


def build_online_overlay_lesson(
    *,
    feature_row: Mapping[str, Any],
    expected_feature_row_sha256: str,
    maturity_session: str,
    active_log_edge_10bps: float,
    label_evidence_sha256: str,
) -> dict[str, Any]:
    """Seal one matured counterfactual result with its decision-time row."""

    feature = validate_online_overlay_feature_row(
        feature_row,
        expected_feature_row_sha256=expected_feature_row_sha256,
    )
    maturity = _iso_date(maturity_session, "lesson maturity session")
    if maturity <= feature["decision_session"]:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Lesson maturity must follow its filing decision session"
        )
    edge_hex = _canonical_float_hex(
        active_log_edge_10bps, "lesson active log edge"
    )
    edge = float.fromhex(edge_hex)
    body = {
        "schema_version": LESSON_ROW_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "accession_number": feature["accession_number"],
        "decision_session": feature["decision_session"],
        "maturity_session": maturity,
        "feature_row": feature,
        "feature_row_sha256": feature["feature_row_sha256"],
        "trainable": bool(feature["fit_eligible"]),
        "binary_cash_win_target": int(edge > POSITIVE_EDGE_TOLERANCE),
        "active_log_edge_10bps_hex": edge_hex,
        "label_evidence_sha256": _sha256(
            label_evidence_sha256, "lesson label evidence hash"
        ),
    }
    return {**body, "lesson_row_sha256": canonical_sha256(body)}


def validate_online_overlay_lesson(
    row: Mapping[str, Any],
    *,
    expected_lesson_row_sha256: str,
) -> dict[str, Any]:
    """Validate one immutable matured lesson and its embedded feature row."""

    value = dict(_mapping(row, "lesson row"))
    if set(value) != _LESSON_ROW_KEYS:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Lesson row keys changed"
        )
    if (
        value["schema_version"] != LESSON_ROW_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"] != CONTRACT_SHA256
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Lesson row experiment identity changed"
        )
    feature_hash = _sha256(
        value["feature_row_sha256"], "lesson feature row hash"
    )
    feature = validate_online_overlay_feature_row(
        _mapping(value["feature_row"], "lesson feature row"),
        expected_feature_row_sha256=feature_hash,
    )
    if (
        value["accession_number"] != feature["accession_number"]
        or value["decision_session"] != feature["decision_session"]
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Lesson identity differs from its feature row"
        )
    decision = _iso_date(value["decision_session"], "lesson decision session")
    maturity = _iso_date(value["maturity_session"], "lesson maturity session")
    if maturity <= decision:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Lesson maturity must follow its filing decision session"
        )
    trainable = _boolean(value["trainable"], "lesson trainable")
    if trainable != bool(feature["fit_eligible"]):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Lesson trainability differs from its immutable feature row"
        )
    target = value["binary_cash_win_target"]
    if isinstance(target, bool) or target not in {0, 1}:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Lesson binary target must be zero or one"
        )
    edge = _decode_float_hex(
        value["active_log_edge_10bps_hex"], "lesson active log edge"
    )
    if target != int(edge > POSITIVE_EDGE_TOLERANCE):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Lesson binary target differs from its frozen edge definition"
        )
    _sha256(value["label_evidence_sha256"], "lesson label evidence hash")
    _hash_body(
        value,
        "lesson_row_sha256",
        expected_sha256=expected_lesson_row_sha256,
        location="lesson row",
    )
    return copy.deepcopy(value)


def evaluate_online_overlay_gate(
    *,
    cash_win_probability: float,
    expected_incremental_10bps_log_edge: float,
) -> dict[str, Any]:
    """Apply the two inclusive, non-searchable preregistered thresholds."""

    probability_hex = _canonical_float_hex(
        cash_win_probability, "cash-win probability"
    )
    edge_hex = _canonical_float_hex(
        expected_incremental_10bps_log_edge,
        "expected incremental log edge",
    )
    probability = float.fromhex(probability_hex)
    edge = float.fromhex(edge_hex)
    if not 0.0 <= probability <= 1.0:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Cash-win probability must lie in [0, 1]"
        )
    probability_passed = probability >= PROBABILITY_GATE
    edge_passed = edge >= EXPECTED_EDGE_GATE
    body = {
        "schema_version": GATE_AUDIT_SCHEMA_VERSION,
        "probability_hex": probability_hex,
        "expected_incremental_10bps_log_edge_hex": edge_hex,
        "probability_gate_hex": PROBABILITY_GATE.hex(),
        "expected_edge_gate_hex": EXPECTED_EDGE_GATE.hex(),
        "probability_gate_passed": probability_passed,
        "expected_edge_gate_passed": edge_passed,
        "overlay_gate_passed": probability_passed and edge_passed,
    }
    return {**body, "gate_audit_sha256": canonical_sha256(body)}


def _candidate_sha256(arm: str) -> str:
    return canonical_sha256(
        {
            "schema_version": LEARNER_ADAPTER_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "arm": arm,
            "vector_field": ARM_VECTOR_FIELDS[arm],
            "head_variant": ARM_HEAD_VARIANTS[arm],
            "feature_names": list(FEATURES),
            "minimum_training_rows": MINIMUM_TRAINING_ROWS,
            "minimum_rows_per_binary_class": MINIMUM_CLASS_ROWS,
            "probability_gate_hex": PROBABILITY_GATE.hex(),
            "expected_edge_gate_hex": EXPECTED_EDGE_GATE.hex(),
            "learner_primitive_sha256": SOURCE_PINS[
                "sec_filing_gemma_learner"
            ],
        }
    )


def _validated_lessons(
    lesson_rows: Sequence[Mapping[str, Any]],
    expected_lesson_row_sha256s: Sequence[str],
    *,
    as_of_session: str,
) -> list[dict[str, Any]]:
    if isinstance(lesson_rows, (str, bytes)) or not isinstance(
        lesson_rows, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Matured lessons must be a sequence"
        )
    if isinstance(expected_lesson_row_sha256s, (str, bytes)) or not isinstance(
        expected_lesson_row_sha256s, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Expected lesson hashes must be a sequence"
        )
    pins = [
        _sha256(item, f"expected lesson hash {index}")
        for index, item in enumerate(expected_lesson_row_sha256s)
    ]
    if len(pins) != len(set(pins)) or len(pins) != len(lesson_rows):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Expected lesson hashes must uniquely pin every supplied lesson"
        )
    by_hash: dict[str, Mapping[str, Any]] = {}
    for raw in lesson_rows:
        value = _mapping(raw, "matured lesson")
        observed = _sha256(
            value.get("lesson_row_sha256"), "matured lesson hash"
        )
        if observed in by_hash:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Duplicate matured lesson supplied"
            )
        by_hash[observed] = value
    if set(by_hash) != set(pins):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Expected lesson hashes do not match supplied lessons"
        )
    validated = [
        validate_online_overlay_lesson(
            by_hash[pin],
            expected_lesson_row_sha256=pin,
        )
        for pin in pins
    ]
    seen_accessions: set[str] = set()
    seen_features: set[str] = set()
    for lesson in validated:
        if lesson["maturity_session"] > as_of_session:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Future outcome lesson reached a causal fit checkpoint"
            )
        if lesson["accession_number"] in seen_accessions:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Multiple lessons refer to one filing accession"
            )
        if lesson["feature_row_sha256"] in seen_features:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Multiple lessons refer to one immutable feature row"
            )
        seen_accessions.add(lesson["accession_number"])
        seen_features.add(lesson["feature_row_sha256"])
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for lesson in validated:
        by_decision.setdefault(lesson["decision_session"], []).append(
            lesson
        )
    chronological: list[dict[str, Any]] = []
    for lesson_decision in sorted(by_decision):
        same_session = by_decision[lesson_decision]
        if len(same_session) > 1 and any(
            item["feature_row"]["acceptance_datetime"] is None
            for item in same_session
        ):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Same-session lesson order is ambiguous without exact "
                "acceptance timestamps"
            )
        if len(same_session) > 1:
            same_session.sort(
                key=lambda item: (
                    item["feature_row"]["acceptance_datetime"],
                    item["accession_number"],
                    item["lesson_row_sha256"],
                )
            )
        chronological.extend(same_session)
    return chronological


def _admitted_lesson_entry(
    lesson: Mapping[str, Any],
) -> dict[str, Any]:
    feature = lesson["feature_row"]
    return {
        "lesson_row_sha256": lesson["lesson_row_sha256"],
        "feature_row_sha256": lesson["feature_row_sha256"],
        "accession_number": lesson["accession_number"],
        "decision_session": lesson["decision_session"],
        "acceptance_datetime": feature["acceptance_datetime"],
        "maturity_session": lesson["maturity_session"],
        "trainable": lesson["trainable"],
    }


def _training_member_entry(
    lesson: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    vector = _select_vector(lesson["feature_row"], arm)
    if vector is None:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Trainable lesson has no immutable feature vector"
        )
    return {
        **_admitted_lesson_entry(lesson),
        "binary_cash_win_target": lesson["binary_cash_win_target"],
        "active_log_edge_10bps_hex": lesson[
            "active_log_edge_10bps_hex"
        ],
        "feature_values_hex": [
            _canonical_float_hex(value, f"{arm} training feature")
            for value in vector
        ],
    }


def _fit_state_from_members(
    *,
    training_members: Sequence[Mapping[str, Any]],
    arm: str,
    as_of_session: str,
    candidate_sha256: str,
    training_membership_sha256: str,
) -> dict[str, Any]:
    matrix = [
        [
            _decode_float_hex(
                value,
                f"training member {row_index} feature {column_index}",
            )
            for column_index, value in enumerate(
                member["feature_values_hex"]
            )
        ]
        for row_index, member in enumerate(training_members)
    ]
    binary = [
        member["binary_cash_win_target"]
        for member in training_members
    ]
    edge = [
        _decode_float_hex(
            member["active_log_edge_10bps_hex"],
            "training member active log edge",
        )
        for member in training_members
    ]
    maximum_maturity = max(
        str(member["maturity_session"]) for member in training_members
    )
    metadata = {
        "candidate_sha256": candidate_sha256,
        "head_variant": ARM_HEAD_VARIANTS[arm],
        "fold_id": f"continuous_expanding:{as_of_session}:{arm}",
        "train_label_maturity_through": as_of_session,
        "training_set_sha256": training_membership_sha256,
        "training_row_count": len(training_members),
        "maximum_training_label_maturity_session": maximum_maturity,
        "feature_schema_sha256": canonical_sha256(list(FEATURES)),
    }
    try:
        learner = SecFilingGemmaTwoHeadLearner().fit(
            matrix,
            binary,
            edge,
            feature_names=FEATURES,
            fit_metadata=metadata,
        )
    except SecFilingGemmaLearnerError as exc:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Frozen two-head learner rejected the causal training set"
        ) from exc
    return learner.to_state()


def build_online_overlay_fit_audit(
    *,
    matured_lessons: Sequence[Mapping[str, Any]],
    expected_lesson_row_sha256s: Sequence[str],
    arm: str,
    as_of_session: str,
) -> dict[str, Any]:
    """Build one causal expanding fit checkpoint on any market session."""

    selected_arm = _arm(arm)
    as_of = _iso_date(as_of_session, "fit as-of session")
    lessons = _validated_lessons(
        matured_lessons,
        expected_lesson_row_sha256s,
        as_of_session=as_of,
    )
    trainable = [item for item in lessons if item["trainable"]]
    positive_count = sum(
        int(item["binary_cash_win_target"]) for item in trainable
    )
    negative_count = len(trainable) - positive_count
    ready = (
        len(trainable) >= MINIMUM_TRAINING_ROWS
        and positive_count >= MINIMUM_CLASS_ROWS
        and negative_count >= MINIMUM_CLASS_ROWS
    )
    admitted = [_admitted_lesson_entry(item) for item in lessons]
    members = [
        _training_member_entry(item, arm=selected_arm)
        for item in trainable
    ]
    membership_sha256 = canonical_sha256(members)
    candidate_sha256 = _candidate_sha256(selected_arm)
    fit_state: dict[str, Any] | None = None
    if ready:
        fit_state = _fit_state_from_members(
            training_members=members,
            arm=selected_arm,
            as_of_session=as_of,
            candidate_sha256=candidate_sha256,
            training_membership_sha256=membership_sha256,
        )
    body = {
        "schema_version": FIT_AUDIT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "as_of_session": as_of,
        "arm": selected_arm,
        "candidate_sha256": candidate_sha256,
        "head_variant": ARM_HEAD_VARIANTS[selected_arm],
        "feature_names": list(FEATURES),
        "feature_schema_sha256": canonical_sha256(list(FEATURES)),
        "learner_primitive_sha256": SOURCE_PINS[
            "sec_filing_gemma_learner"
        ],
        "supplied_matured_lesson_count": len(lessons),
        "audit_only_lesson_count": len(lessons) - len(trainable),
        "training_row_count": len(trainable),
        "training_positive_count": positive_count,
        "training_negative_count": negative_count,
        "minimum_training_rows": MINIMUM_TRAINING_ROWS,
        "minimum_rows_per_binary_class": MINIMUM_CLASS_ROWS,
        "learner_ready": ready,
        "admitted_lessons": admitted,
        "admitted_lessons_sha256": canonical_sha256(admitted),
        "training_members": members,
        "training_membership_sha256": membership_sha256,
        "fit_state": fit_state,
        "fit_state_sha256": (
            None if fit_state is None else fit_state["state_sha256"]
        ),
    }
    return {**body, "fit_audit_sha256": canonical_sha256(body)}


def _strict_count(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} must be a nonnegative integer"
        )
    return value


def _validate_acceptance(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 14
        or not value.isascii()
        or not value.isdigit()
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            f"{location} is invalid"
        )
    return value


def _validate_admitted_lessons(
    value: Any,
    *,
    as_of_session: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit admitted lessons must be a list"
        )
    result: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_accessions: set[str] = set()
    seen_features: set[str] = set()
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for index, raw in enumerate(value):
        item = dict(_mapping(raw, f"admitted lesson {index}"))
        if set(item) != _ADMITTED_LESSON_KEYS:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit admitted lesson keys changed"
            )
        lesson_hash = _sha256(
            item["lesson_row_sha256"],
            f"admitted lesson {index} hash",
        )
        feature_hash = _sha256(
            item["feature_row_sha256"],
            f"admitted lesson {index} feature hash",
        )
        accession = item["accession_number"]
        if not isinstance(accession, str) or not accession:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit admitted lesson accession is invalid"
            )
        decision = _iso_date(
            item["decision_session"],
            f"admitted lesson {index} decision session",
        )
        acceptance = _validate_acceptance(
            item["acceptance_datetime"],
            f"admitted lesson {index} acceptance timestamp",
        )
        maturity = _iso_date(
            item["maturity_session"],
            f"admitted lesson {index} maturity session",
        )
        if not decision < maturity <= as_of_session:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit admitted lesson is not causally mature"
            )
        _boolean(item["trainable"], f"admitted lesson {index} trainable")
        if (
            lesson_hash in seen_hashes
            or accession in seen_accessions
            or feature_hash in seen_features
        ):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit admitted lesson membership contains duplicates"
            )
        seen_hashes.add(lesson_hash)
        seen_accessions.add(accession)
        seen_features.add(feature_hash)
        normalized = {
            **item,
            "decision_session": decision,
            "acceptance_datetime": acceptance,
            "maturity_session": maturity,
        }
        result.append(normalized)
        by_decision.setdefault(decision, []).append(normalized)
    for same_session in by_decision.values():
        if len(same_session) > 1 and any(
            item["acceptance_datetime"] is None for item in same_session
        ):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Same-session lesson order is ambiguous without exact "
                "acceptance timestamps"
            )
    expected: list[dict[str, Any]] = []
    for lesson_decision in sorted(by_decision):
        same_session = list(by_decision[lesson_decision])
        if len(same_session) > 1:
            same_session.sort(
                key=lambda item: (
                    item["acceptance_datetime"],
                    item["accession_number"],
                    item["lesson_row_sha256"],
                )
            )
        expected.extend(same_session)
    if result != expected:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit admitted lesson order changed"
        )
    return result


def _validate_training_members(
    value: Any,
    *,
    admitted_lessons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit training members must be a list"
        )
    expected_admitted = [
        item for item in admitted_lessons if item["trainable"]
    ]
    if len(value) != len(expected_admitted):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit training membership count changed"
        )
    result: list[dict[str, Any]] = []
    for index, (raw, admitted) in enumerate(
        zip(value, expected_admitted, strict=True)
    ):
        item = dict(_mapping(raw, f"training member {index}"))
        if set(item) != _TRAINING_MEMBER_KEYS:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit training member keys changed"
            )
        admitted_projection = {
            key: item[key] for key in _ADMITTED_LESSON_KEYS
        }
        if admitted_projection != dict(admitted):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit training member differs from admitted membership"
            )
        target = item["binary_cash_win_target"]
        if isinstance(target, bool) or target not in {0, 1}:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit training member binary target changed"
            )
        edge = _decode_float_hex(
            item["active_log_edge_10bps_hex"],
            f"training member {index} active log edge",
        )
        if target != int(edge > POSITIVE_EDGE_TOLERANCE):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit training member target differs from edge"
            )
        vector = item["feature_values_hex"]
        if not isinstance(vector, list) or len(vector) != len(FEATURES):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit training member feature vector changed"
            )
        for column, encoded in enumerate(vector):
            _decode_float_hex(
                encoded,
                f"training member {index} feature {column}",
            )
        result.append(item)
    return result


def _validate_fit_audit(
    fit_audit: Mapping[str, Any],
    *,
    expected_fit_audit_sha256: str,
    replay_fit: bool,
) -> dict[str, Any]:
    value = dict(_mapping(fit_audit, "fit audit"))
    if set(value) != _FIT_AUDIT_KEYS:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit audit keys changed"
        )
    if (
        value["schema_version"] != FIT_AUDIT_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["contract_sha256"] != CONTRACT_SHA256
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit audit experiment identity changed"
        )
    as_of = _iso_date(value["as_of_session"], "fit as-of session")
    selected_arm = _arm(value["arm"])
    candidate_sha256 = _sha256(
        value["candidate_sha256"], "fit candidate hash"
    )
    if candidate_sha256 != _candidate_sha256(selected_arm):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit candidate identity changed"
        )
    if value["head_variant"] != ARM_HEAD_VARIANTS[selected_arm]:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit learner head variant changed"
        )
    if (
        value["feature_names"] != list(FEATURES)
        or value["feature_schema_sha256"]
        != canonical_sha256(list(FEATURES))
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit feature schema changed"
        )
    if value["learner_primitive_sha256"] != SOURCE_PINS[
        "sec_filing_gemma_learner"
    ]:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit learner primitive identity changed"
        )
    admitted = _validate_admitted_lessons(
        value["admitted_lessons"],
        as_of_session=as_of,
    )
    if value["admitted_lessons_sha256"] != canonical_sha256(admitted):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit admitted lesson membership checksum changed"
        )
    members = _validate_training_members(
        value["training_members"],
        admitted_lessons=admitted,
    )
    if value["training_membership_sha256"] != canonical_sha256(members):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit training membership checksum changed"
        )
    supplied_count = _strict_count(
        value["supplied_matured_lesson_count"],
        "fit supplied matured lesson count",
    )
    audit_only_count = _strict_count(
        value["audit_only_lesson_count"],
        "fit audit-only lesson count",
    )
    row_count = _strict_count(
        value["training_row_count"], "fit training row count"
    )
    positive_count = _strict_count(
        value["training_positive_count"],
        "fit training positive count",
    )
    negative_count = _strict_count(
        value["training_negative_count"],
        "fit training negative count",
    )
    if (
        supplied_count != len(admitted)
        or row_count != len(members)
        or audit_only_count != len(admitted) - len(members)
        or positive_count
        != sum(item["binary_cash_win_target"] for item in members)
        or negative_count != row_count - positive_count
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit membership counts changed"
        )
    if (
        value["minimum_training_rows"] != MINIMUM_TRAINING_ROWS
        or value["minimum_rows_per_binary_class"] != MINIMUM_CLASS_ROWS
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit readiness thresholds changed"
        )
    ready = (
        row_count >= MINIMUM_TRAINING_ROWS
        and positive_count >= MINIMUM_CLASS_ROWS
        and negative_count >= MINIMUM_CLASS_ROWS
    )
    if (
        not isinstance(value["learner_ready"], bool)
        or value["learner_ready"] != ready
    ):
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Fit learner readiness changed"
        )
    fit_state = value["fit_state"]
    fit_state_hash = value["fit_state_sha256"]
    if not ready:
        if fit_state is not None or fit_state_hash is not None:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Unready fit audit exposes learner state"
            )
    else:
        state = dict(_mapping(fit_state, "fit learner state"))
        observed_state_hash = _sha256(
            fit_state_hash, "fit learner state hash"
        )
        try:
            learner = SecFilingGemmaTwoHeadLearner.from_state(state)
        except SecFilingGemmaLearnerError as exc:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit learner state is invalid"
            ) from exc
        if learner.model_sha256 != observed_state_hash:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit learner state hash changed"
            )
        metadata = learner.fit_metadata
        assert metadata is not None
        maximum_maturity = max(
            item["maturity_session"] for item in members
        )
        if (
            metadata.candidate_sha256 != candidate_sha256
            or metadata.head_variant != ARM_HEAD_VARIANTS[selected_arm]
            or metadata.fold_id
            != f"continuous_expanding:{as_of}:{selected_arm}"
            or metadata.train_label_maturity_through != as_of
            or metadata.training_set_sha256
            != value["training_membership_sha256"]
            or metadata.training_row_count != row_count
            or metadata.maximum_training_label_maturity_session
            != maximum_maturity
            or metadata.feature_schema_sha256
            != value["feature_schema_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Fit learner metadata differs from its audit membership"
            )
        if replay_fit:
            replayed = _fit_state_from_members(
                training_members=members,
                arm=selected_arm,
                as_of_session=as_of,
                candidate_sha256=candidate_sha256,
                training_membership_sha256=value[
                    "training_membership_sha256"
                ],
            )
            if replayed != state:
                raise SecGemmaOnlineRiskOverlayLearnerError(
                    "Fit learner state differs from deterministic replay"
                )
    _hash_body(
        value,
        "fit_audit_sha256",
        expected_sha256=expected_fit_audit_sha256,
        location="fit audit",
    )
    return copy.deepcopy(value)


def validate_online_overlay_fit_audit(
    fit_audit: Mapping[str, Any],
    *,
    expected_fit_audit_sha256: str,
) -> dict[str, Any]:
    """Self-validate and deterministically replay one fit checkpoint."""

    return _validate_fit_audit(
        fit_audit,
        expected_fit_audit_sha256=expected_fit_audit_sha256,
        replay_fit=True,
    )


def build_online_overlay_prediction_from_fit(
    *,
    feature_row: Mapping[str, Any],
    expected_feature_row_sha256: str,
    fit_audit: Mapping[str, Any],
    expected_fit_audit_sha256: str,
) -> dict[str, Any]:
    """Predict from one sealed fit without invoking the fitting routine."""

    current = validate_online_overlay_feature_row(
        feature_row,
        expected_feature_row_sha256=expected_feature_row_sha256,
    )
    fit = _validate_fit_audit(
        fit_audit,
        expected_fit_audit_sha256=expected_fit_audit_sha256,
        replay_fit=False,
    )
    decision = current["decision_session"]
    if fit["as_of_session"] > decision:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Prediction cannot consume a future fit checkpoint"
        )
    selected_arm = fit["arm"]
    ready = fit["learner_ready"]
    fitted_prediction_available = bool(
        ready and current["prediction_available"]
    )
    gate_audit: dict[str, Any] | None = None
    if fitted_prediction_available:
        state = fit["fit_state"]
        assert isinstance(state, Mapping)
        vector = _select_vector(current, selected_arm)
        assert vector is not None
        try:
            learner = SecFilingGemmaTwoHeadLearner.from_state(state)
            components = learner.predict_components(
                vector
            )
        except SecFilingGemmaLearnerError as exc:
            raise SecGemmaOnlineRiskOverlayLearnerError(
                "Frozen two-head learner could not reproduce its prediction"
            ) from exc
        probability = float(
            components["cash_win_probability_10bps"][0]
        )
        expected_edge = float(
            components["expected_active_log_edge_10bps"][0]
        )
        gate_audit = evaluate_online_overlay_gate(
            cash_win_probability=probability,
            expected_incremental_10bps_log_edge=expected_edge,
        )
    prediction_body = {
        "schema_version": PREDICTION_AUDIT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "decision_session": decision,
        "accession_number": current["accession_number"],
        "acceptance_datetime": current["acceptance_datetime"],
        "feature_row_sha256": current["feature_row_sha256"],
        "arm": selected_arm,
        "fit_as_of_session": fit["as_of_session"],
        "frozen_fit_reuse": fit["as_of_session"] < decision,
        "learner_ready": ready,
        "current_feature_available": bool(
            current["prediction_available"]
        ),
        "prediction_available": bool(current["prediction_available"]),
        "fitted_prediction_available": fitted_prediction_available,
        "action_available": fitted_prediction_available,
        "raw_gate_pass": (
            False
            if gate_audit is None
            else bool(gate_audit["overlay_gate_passed"])
        ),
        "gate_schedule_overlay": (
            False
            if gate_audit is None
            else bool(gate_audit["overlay_gate_passed"])
        ),
        "gate_audit": gate_audit,
        "gate_audit_sha256": (
            None if gate_audit is None else gate_audit["gate_audit_sha256"]
        ),
        "fit_audit": copy.deepcopy(fit),
        "fit_audit_sha256": fit["fit_audit_sha256"],
    }
    return {
        **prediction_body,
        "prediction_row_sha256": canonical_sha256(prediction_body),
    }


def validate_online_overlay_prediction_from_fit(
    prediction_row: Mapping[str, Any],
    *,
    expected_prediction_row_sha256: str,
    feature_row: Mapping[str, Any],
    expected_feature_row_sha256: str,
    fit_audit: Mapping[str, Any],
    expected_fit_audit_sha256: str,
) -> dict[str, Any]:
    """Rebuild one prediction from its exact feature and fit checkpoints."""

    value = dict(_mapping(prediction_row, "prediction row"))
    if set(value) != _PREDICTION_ROW_KEYS:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Prediction row keys changed"
        )
    rebuilt = build_online_overlay_prediction_from_fit(
        feature_row=feature_row,
        expected_feature_row_sha256=expected_feature_row_sha256,
        fit_audit=fit_audit,
        expected_fit_audit_sha256=expected_fit_audit_sha256,
    )
    if value != rebuilt:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Prediction row differs from deterministic fit reuse"
        )
    _hash_body(
        value,
        "prediction_row_sha256",
        expected_sha256=expected_prediction_row_sha256,
        location="prediction row",
    )
    return copy.deepcopy(value)


def build_online_overlay_prediction(
    *,
    feature_row: Mapping[str, Any],
    expected_feature_row_sha256: str,
    matured_lessons: Sequence[Mapping[str, Any]],
    expected_lesson_row_sha256s: Sequence[str],
    arm: str,
    decision_session: str,
) -> dict[str, Any]:
    """Compatibility composition of one causal fit and one prediction."""

    decision = _iso_date(decision_session, "prediction decision session")
    current = validate_online_overlay_feature_row(
        feature_row,
        expected_feature_row_sha256=expected_feature_row_sha256,
    )
    if current["decision_session"] != decision:
        raise SecGemmaOnlineRiskOverlayLearnerError(
            "Prediction decision session differs from its feature row"
        )
    fit = build_online_overlay_fit_audit(
        matured_lessons=matured_lessons,
        expected_lesson_row_sha256s=expected_lesson_row_sha256s,
        arm=arm,
        as_of_session=decision,
    )
    return build_online_overlay_prediction_from_fit(
        feature_row=current,
        expected_feature_row_sha256=current["feature_row_sha256"],
        fit_audit=fit,
        expected_fit_audit_sha256=fit["fit_audit_sha256"],
    )


__all__ = [
    "ARM_HEAD_VARIANTS",
    "ARM_VECTOR_FIELDS",
    "FEATURE_ARMS",
    "FIT_AUDIT_SCHEMA_VERSION",
    "GATE_AUDIT_SCHEMA_VERSION",
    "LEARNER_ADAPTER_SCHEMA_VERSION",
    "LESSON_ROW_SCHEMA_VERSION",
    "PREDICTION_AUDIT_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayLearnerError",
    "build_online_overlay_fit_audit",
    "build_online_overlay_lesson",
    "build_online_overlay_prediction",
    "build_online_overlay_prediction_from_fit",
    "evaluate_online_overlay_gate",
    "select_online_overlay_feature_arm",
    "validate_online_overlay_feature_row",
    "validate_online_overlay_fit_audit",
    "validate_online_overlay_lesson",
    "validate_online_overlay_prediction_from_fit",
]
