from __future__ import annotations

"""Deterministic two-state regime model for the frozen AAPL consensus system.

The HMM is fitted only on complete emissions, while gaps split the chronology
into independent sequences.  Prediction is causal: only the forward filter is
available publicly, and an incomplete row resets the next observation to the
fitted initial distribution.
"""

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


MODEL_TYPE = "aapl_regime_consensus_two_state_gaussian_hmm"
OUTCOME_MODEL_TYPE = "aapl_regime_consensus_outcome_head"
STATE_SCHEMA_VERSION = 1


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
        number = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{location} must be a canonical hexadecimal float") from exc
    if not math.isfinite(number) or number.hex() != value:
        raise ValueError(f"{location} must be a canonical finite hexadecimal float")
    return number


def _encode_vector(values: Sequence[float]) -> list[str]:
    return [_float_hex(value) for value in np.asarray(values, dtype=float)]


def _decode_vector(values: Any, location: str, length: int) -> np.ndarray:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{location} must contain exactly {length} values")
    return np.asarray(
        [_decode_float_hex(item, f"{location}[{index}]") for index, item in enumerate(values)],
        dtype=float,
    )


def _encode_matrix(values: np.ndarray) -> list[list[str]]:
    matrix = np.asarray(values, dtype=float)
    return [_encode_vector(row) for row in matrix]


def _decode_matrix(values: Any, location: str, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(values, list) or len(values) != shape[0]:
        raise ValueError(f"{location} must contain exactly {shape[0]} rows")
    return np.vstack(
        [_decode_vector(row, f"{location}[{index}]", shape[1]) for index, row in enumerate(values)]
    )


@dataclass(frozen=True)
class RegimeHMMConfig:
    n_states: int = 2
    raw_mad_multiplier: float = 1.4826
    raw_scale_floor: float = 1e-6
    raw_z_clip: float = 6.0
    variance_floor: float = 0.0025
    transition_additive: tuple[tuple[float, float], tuple[float, float]] = (
        (5.0, 1.0),
        (1.0, 5.0),
    )
    pi_additive: tuple[float, float] = (1.0, 1.0)
    max_iterations: int = 100
    min_iterations: int = 5
    relative_tolerance: float = 1e-10
    min_complete_emissions: int = 500
    min_effective_occupancy: float = 50.0
    outcome_prior: float = 100.0

    def validate(self) -> None:
        if _canonical_json(asdict(self)) != _canonical_json(asdict(RegimeHMMConfig())):
            raise ValueError("HMM configuration does not match the frozen regime-v1 contract")


@dataclass(frozen=True)
class RegimeFitMetadata:
    training_start_date: str
    training_end_date: str
    maximum_label_maturity_date: str
    training_dates_sha256: str
    binary_target_sha256: str
    edge_target_sha256: str

    def validate(self) -> None:
        parsed: dict[str, date] = {}
        for field_name in (
            "training_start_date",
            "training_end_date",
            "maximum_label_maturity_date",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be an ISO date")
            try:
                parsed_value = date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be an ISO date") from exc
            if parsed_value.isoformat() != value:
                raise ValueError(f"{field_name} must use canonical YYYY-MM-DD form")
            parsed[field_name] = parsed_value
        if parsed["training_start_date"] > parsed["training_end_date"]:
            raise ValueError("training_start_date must not follow training_end_date")
        for field_name in (
            "training_dates_sha256",
            "binary_target_sha256",
            "edge_target_sha256",
        ):
            if not _is_sha256(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")

    @classmethod
    def coerce(cls, value: Mapping[str, Any] | "RegimeFitMetadata") -> "RegimeFitMetadata":
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            _expect_keys(value, set(cls.__dataclass_fields__), "fit metadata")
            try:
                result = cls(**dict(value))
            except TypeError as exc:
                raise ValueError("Could not decode regime fit metadata") from exc
        else:
            raise ValueError("fit_metadata must be a RegimeFitMetadata or mapping")
        result.validate()
        return result


def _config_state(config: RegimeHMMConfig) -> dict[str, Any]:
    return {
        "n_states": config.n_states,
        "raw_mad_multiplier": _float_hex(config.raw_mad_multiplier),
        "raw_scale_floor": _float_hex(config.raw_scale_floor),
        "raw_z_clip": _float_hex(config.raw_z_clip),
        "variance_floor": _float_hex(config.variance_floor),
        "transition_additive": _encode_matrix(np.asarray(config.transition_additive)),
        "pi_additive": _encode_vector(config.pi_additive),
        "max_iterations": config.max_iterations,
        "min_iterations": config.min_iterations,
        "relative_tolerance": _float_hex(config.relative_tolerance),
        "min_complete_emissions": config.min_complete_emissions,
        "min_effective_occupancy": _float_hex(config.min_effective_occupancy),
        "outcome_prior": _float_hex(config.outcome_prior),
    }


def _config_from_state(value: Any) -> RegimeHMMConfig:
    if not isinstance(value, Mapping):
        raise ValueError("HMM config must be an object")
    _expect_keys(value, set(asdict(RegimeHMMConfig())), "config")
    config = RegimeHMMConfig(
        n_states=_strict_int(value["n_states"], "n_states", minimum=1),
        raw_mad_multiplier=_decode_float_hex(value["raw_mad_multiplier"], "raw_mad_multiplier"),
        raw_scale_floor=_decode_float_hex(value["raw_scale_floor"], "raw_scale_floor"),
        raw_z_clip=_decode_float_hex(value["raw_z_clip"], "raw_z_clip"),
        variance_floor=_decode_float_hex(value["variance_floor"], "variance_floor"),
        transition_additive=tuple(
            tuple(row) for row in _decode_matrix(value["transition_additive"], "transition_additive", (2, 2))
        ),
        pi_additive=tuple(_decode_vector(value["pi_additive"], "pi_additive", 2)),
        max_iterations=_strict_int(value["max_iterations"], "max_iterations", minimum=1),
        min_iterations=_strict_int(value["min_iterations"], "min_iterations", minimum=1),
        relative_tolerance=_decode_float_hex(value["relative_tolerance"], "relative_tolerance"),
        min_complete_emissions=_strict_int(
            value["min_complete_emissions"], "min_complete_emissions", minimum=1
        ),
        min_effective_occupancy=_decode_float_hex(
            value["min_effective_occupancy"], "min_effective_occupancy"
        ),
        outcome_prior=_decode_float_hex(value["outcome_prior"], "outcome_prior"),
    )
    config.validate()
    return config


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    array = np.asarray(values, dtype=float)
    maximum = np.max(array, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(array - maximum), axis=axis, keepdims=True))
    if axis is None:
        return float(result.reshape(-1)[0])
    return np.squeeze(result, axis=axis)


def _emission_log_probability(
    observations: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    differences = observations[:, None, :] - means[None, :, :]
    return -0.5 * (
        observations.shape[1] * math.log(2.0 * math.pi)
        + np.sum(np.log(variances), axis=1)[None, :]
        + np.sum((differences * differences) / variances[None, :, :], axis=2)
    )


def _forward_backward(
    observations: np.ndarray,
    pi: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    emission = _emission_log_probability(observations, means, variances)
    log_pi = np.log(pi)
    log_transition = np.log(transition)
    length = len(observations)
    alpha = np.empty((length, 2), dtype=float)
    alpha[0] = log_pi + emission[0]
    for position in range(1, length):
        alpha[position] = emission[position] + _logsumexp(
            alpha[position - 1][:, None] + log_transition, axis=0
        )
    likelihood = float(_logsumexp(alpha[-1]))
    beta = np.zeros((length, 2), dtype=float)
    for position in range(length - 2, -1, -1):
        beta[position] = _logsumexp(
            log_transition + emission[position + 1][None, :] + beta[position + 1][None, :],
            axis=1,
        )
    gamma_log = alpha + beta - likelihood
    gamma = np.exp(gamma_log)
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((2, 2), dtype=float)
    for position in range(length - 1):
        xi_log = (
            alpha[position][:, None]
            + log_transition
            + emission[position + 1][None, :]
            + beta[position + 1][None, :]
            - likelihood
        )
        xi = np.exp(xi_log)
        xi_sum += xi / xi.sum()
    return likelihood, gamma, xi_sum


def _sequence_slices(complete: np.ndarray) -> list[slice]:
    slices: list[slice] = []
    start: int | None = None
    for index, available in enumerate(np.asarray(complete, dtype=bool)):
        if available and start is None:
            start = index
        if start is not None and (not available or index == len(complete) - 1):
            stop = index if not available else index + 1
            slices.append(slice(start, stop))
            start = None
    return slices


def _oriented_order(
    pi: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    orientations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stress = np.mean(means * orientations[None, :], axis=1)
    swap = bool(stress[0] > stress[1])
    if stress[0] == stress[1]:
        first_key = tuple(means[0])
        second_key = tuple(means[1])
        swap = first_key > second_key
    if not swap:
        return pi, transition, means, variances
    order = np.asarray([1, 0])
    return (
        pi[order],
        transition[np.ix_(order, order)],
        means[order],
        variances[order],
    )


def _likelihood_decreased_beyond_guard(previous: float, current: float) -> bool:
    """Frozen numerical-dust allowance for the Baum-Welch monotonicity check."""

    return current < previous - 1e-9 * (1.0 + abs(previous))


def _regularized_log_objective(
    raw_log_likelihood: float,
    pi: np.ndarray,
    transition: np.ndarray,
    config: RegimeHMMConfig,
) -> float:
    """Observed-data log likelihood plus the exact additive-count log prior.

    ``_maximization`` adds the frozen pseudo-count arrays to the expected
    initial-state and transition counts.  That M-step therefore maximizes this
    MAP objective, not raw observed-data likelihood by itself.
    """

    initial_prior = np.asarray(config.pi_additive, dtype=float)
    transition_prior = np.asarray(config.transition_additive, dtype=float)
    return float(
        raw_log_likelihood
        + np.sum(initial_prior * np.log(pi))
        + np.sum(transition_prior * np.log(transition))
    )


class RegimeOutcomeHead:
    """Shrinkage outcome estimates conditional on filtered regime probability."""

    def __init__(self, config: RegimeHMMConfig | None = None) -> None:
        self.config = config or RegimeHMMConfig()
        self.config.validate()
        self.sample_count = 0
        self.positive_count = 0
        self.effective_occupancy = np.empty(0, dtype=float)
        self.global_probability = math.nan
        self.global_edge = math.nan
        self.state_probabilities = np.empty(0, dtype=float)
        self.state_edges = np.empty(0, dtype=float)

    @property
    def is_fitted(self) -> bool:
        return self.state_probabilities.shape == (2,) and self.state_edges.shape == (2,)

    def fit(
        self,
        filtered_state_probabilities: Any,
        cash_beats_long_10bps: Sequence[float],
        edge_10bps: Sequence[float],
    ) -> "RegimeOutcomeHead":
        probabilities = np.asarray(filtered_state_probabilities, dtype=float)
        binary = np.asarray(cash_beats_long_10bps, dtype=float)
        edge = np.asarray(edge_10bps, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("Filtered state probabilities must have shape (rows, 2)")
        if binary.shape != (len(probabilities),) or edge.shape != (len(probabilities),):
            raise ValueError("Outcome targets must align with filtered state probabilities")
        if np.isinf(probabilities).any() or np.isinf(binary).any() or np.isinf(edge).any():
            raise ValueError("Outcome inputs must not contain infinities")
        if not np.array_equal(np.isfinite(binary), np.isfinite(edge)):
            raise ValueError("Binary and edge outcome availability must match")
        available = np.isfinite(binary) & np.isfinite(probabilities).all(axis=1)
        if not np.isin(binary[np.isfinite(binary)], (0.0, 1.0)).all():
            raise ValueError("cash_beats_long_10bps must be binary zero/one where available")
        state = probabilities[available]
        binary_available = binary[available]
        edge_available = edge[available]
        if len(state) < self.config.min_complete_emissions:
            raise ValueError(
                f"Outcome head requires at least {self.config.min_complete_emissions} labeled rows"
            )
        if ((state < 0.0) | (state > 1.0)).any() or not np.allclose(
            state.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Filtered state probabilities must be normalized")
        positive_count = int(binary_available.sum())
        if not 0 < positive_count < len(binary_available):
            raise ValueError("Outcome training requires both binary classes")
        occupancy = state.sum(axis=0)
        if (occupancy < self.config.min_effective_occupancy).any():
            raise ValueError("Outcome state effective occupancy is below the frozen minimum")
        global_probability = float((positive_count + 1.0) / (len(binary_available) + 2.0))
        global_edge = float(edge_available.mean())
        prior = self.config.outcome_prior
        state_probability = (
            state.T @ binary_available + prior * global_probability
        ) / (occupancy + prior)
        state_edge = (state.T @ edge_available + prior * global_edge) / (occupancy + prior)
        self.sample_count = int(len(state))
        self.positive_count = positive_count
        self.effective_occupancy = occupancy
        self.global_probability = global_probability
        self.global_edge = global_edge
        self.state_probabilities = state_probability
        self.state_edges = state_edge
        self._validate_fitted_semantics()
        return self

    def predict_components(self, filtered_state_probabilities: Any) -> dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Regime outcome head is not fitted")
        state = np.asarray(filtered_state_probabilities, dtype=float)
        if state.ndim != 2 or state.shape[1] != 2 or np.isinf(state).any():
            raise ValueError("Filtered state probabilities must have shape (rows, 2) without infinities")
        available = np.isfinite(state).all(axis=1)
        if ((state[available] < 0.0) | (state[available] > 1.0)).any() or not np.allclose(
            state[available].sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Filtered state probabilities must be normalized")
        probability = np.full(len(state), np.nan, dtype=float)
        edge = np.full(len(state), np.nan, dtype=float)
        probability[available] = state[available] @ self.state_probabilities
        edge[available] = state[available] @ self.state_edges
        return {
            "cash_beats_long_probability": probability,
            "expected_edge_10bps": edge,
        }

    def _payload(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Regime outcome head is not fitted")
        return {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "model_type": OUTCOME_MODEL_TYPE,
            "outcome_prior": _float_hex(self.config.outcome_prior),
            "minimum_effective_occupancy": _float_hex(self.config.min_effective_occupancy),
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "effective_occupancy": _encode_vector(self.effective_occupancy),
            "global_probability": _float_hex(self.global_probability),
            "global_edge": _float_hex(self.global_edge),
            "state_probabilities": _encode_vector(self.state_probabilities),
            "state_edges": _encode_vector(self.state_edges),
        }

    def to_state(self) -> dict[str, Any]:
        payload = self._payload()
        return json.loads(_canonical_json({"payload": payload, "payload_sha256": _sha256_json(payload)}))

    @property
    def model_sha256(self) -> str:
        if not self.is_fitted:
            raise RuntimeError("Regime outcome head is not fitted")
        return _sha256_json(self._payload())

    def canonical_json(self) -> str:
        return _canonical_json(self.to_state())

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any], *, config: RegimeHMMConfig | None = None
    ) -> "RegimeOutcomeHead":
        if not isinstance(state, Mapping):
            raise ValueError("Regime outcome state must be an object")
        _expect_keys(state, {"payload", "payload_sha256"}, "outcome state")
        payload = state["payload"]
        if not isinstance(payload, Mapping) or not _is_sha256(state["payload_sha256"]):
            raise ValueError("Regime outcome payload or checksum is invalid")
        if not hmac.compare_digest(_sha256_json(payload), state["payload_sha256"]):
            raise ValueError("Regime outcome state checksum mismatch")
        head = cls(config)
        head._load_payload(payload)
        return head

    def _load_payload(self, payload: Mapping[str, Any]) -> None:
        _expect_keys(
            payload,
            {
                "state_schema_version", "model_type", "outcome_prior",
                "minimum_effective_occupancy", "sample_count", "positive_count",
                "effective_occupancy", "global_probability", "global_edge",
                "state_probabilities", "state_edges",
            },
            "outcome payload",
        )
        if (
            isinstance(payload["state_schema_version"], bool)
            or not isinstance(payload["state_schema_version"], int)
            or payload["state_schema_version"] != STATE_SCHEMA_VERSION
            or payload["model_type"] != OUTCOME_MODEL_TYPE
        ):
            raise ValueError("Unsupported regime outcome state schema or model type")
        if _decode_float_hex(payload["outcome_prior"], "outcome_prior").hex() != self.config.outcome_prior.hex():
            raise ValueError("Outcome prior violates the frozen contract")
        if _decode_float_hex(payload["minimum_effective_occupancy"], "minimum_effective_occupancy").hex() != self.config.min_effective_occupancy.hex():
            raise ValueError("Outcome occupancy threshold violates the frozen contract")
        self.sample_count = _strict_int(
            payload["sample_count"],
            "sample_count",
            minimum=self.config.min_complete_emissions,
        )
        self.positive_count = _strict_int(payload["positive_count"], "positive_count", minimum=1)
        self.effective_occupancy = _decode_vector(payload["effective_occupancy"], "effective_occupancy", 2)
        self.global_probability = _decode_float_hex(payload["global_probability"], "global_probability")
        self.global_edge = _decode_float_hex(payload["global_edge"], "global_edge")
        self.state_probabilities = _decode_vector(payload["state_probabilities"], "state_probabilities", 2)
        self.state_edges = _decode_vector(payload["state_edges"], "state_edges", 2)
        self._validate_fitted_semantics()

    def _validate_fitted_semantics(self) -> None:
        self.config.validate()
        if not self.is_fitted:
            raise ValueError("Regime outcome head is incomplete")
        if (
            self.sample_count < self.config.min_complete_emissions
            or not 0 < self.positive_count < self.sample_count
        ):
            raise ValueError("Regime outcome training counts are invalid")
        if self.effective_occupancy.shape != (2,) or not np.isfinite(self.effective_occupancy).all():
            raise ValueError("Regime outcome occupancy is invalid")
        if (self.effective_occupancy < self.config.min_effective_occupancy).any():
            raise ValueError("Regime outcome occupancy is below the frozen minimum")
        if not math.isclose(float(self.effective_occupancy.sum()), self.sample_count, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Regime outcome occupancy does not match sample count")
        if not 0.0 <= self.global_probability <= 1.0:
            raise ValueError("Regime outcome global probability is invalid")
        if ((self.state_probabilities < 0.0) | (self.state_probabilities > 1.0)).any():
            raise ValueError("Regime outcome state probabilities are invalid")
        if not math.isfinite(self.global_edge) or not np.isfinite(self.state_edges).all():
            raise ValueError("Regime outcome edge estimates are invalid")


class RegimeHeadModel:
    """Frozen deterministic HMM plus a causal regime-conditioned outcome head."""

    def __init__(self, config: RegimeHMMConfig | None = None) -> None:
        self.config = config or RegimeHMMConfig()
        self.config.validate()
        self.head_name = ""
        self.feature_names: tuple[str, ...] = ()
        self.stress_orientations: tuple[int, ...] = ()
        self.fit_metadata: RegimeFitMetadata | None = None
        self.raw_medians = np.empty(0, dtype=float)
        self.raw_scales = np.empty(0, dtype=float)
        self.initial_probabilities = np.empty(0, dtype=float)
        self.transition_matrix = np.empty((0, 0), dtype=float)
        self.means = np.empty((0, 0), dtype=float)
        self.variances = np.empty((0, 0), dtype=float)
        self.effective_occupancy = np.empty(0, dtype=float)
        self.training_row_count = 0
        self.complete_emission_count = 0
        self.sequence_count = 0
        self.iterations = 0
        self.converged = False
        self.final_log_likelihood = math.nan
        self.final_regularized_objective = math.nan
        self.training_terminal_state_probability = np.empty(0, dtype=float)
        self.training_terminal_contiguous = False
        self.outcome_head = RegimeOutcomeHead(self.config)

    @property
    def is_fitted(self) -> bool:
        dimension = len(self.feature_names)
        return (
            self.fit_metadata is not None
            and bool(self.head_name)
            and dimension > 0
            and self.initial_probabilities.shape == (2,)
            and self.transition_matrix.shape == (2, 2)
            and self.means.shape == (2, dimension)
            and self.variances.shape == (2, dimension)
            and self.outcome_head.is_fitted
        )

    @property
    def pi(self) -> np.ndarray:
        return self.initial_probabilities

    @property
    def transition(self) -> np.ndarray:
        return self.transition_matrix

    @property
    def emission_means(self) -> np.ndarray:
        return self.means

    @property
    def emission_variances(self) -> np.ndarray:
        return self.variances

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Regime head model is not fitted")

    @property
    def model_sha256(self) -> str:
        self._require_fitted()
        return _sha256_json(self._payload())

    @staticmethod
    def _coerce_feature_matrix(
        features: Any,
        feature_names: Sequence[str] | None,
        *,
        prediction_names: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        inferred: tuple[str, ...] | None = None
        if hasattr(features, "columns") and hasattr(features, "to_numpy"):
            inferred = tuple(str(column) for column in features.columns)
            matrix = np.asarray(features.to_numpy(dtype=float), dtype=float)
        else:
            matrix = np.asarray(features, dtype=float)
        if matrix.ndim == 1 and prediction_names is not None:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] < 1:
            raise ValueError("Regime features must be a two-dimensional matrix")
        if feature_names is None:
            if inferred is None:
                if prediction_names is None:
                    raise ValueError("feature_names are required for an array feature matrix")
                names = prediction_names
            else:
                names = inferred
        else:
            names = tuple(str(name) for name in feature_names)
            if inferred is not None and names != inferred:
                raise ValueError("Provided feature_names do not match table columns")
        if len(names) != matrix.shape[1] or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Feature names must be unique, non-empty, and match the matrix")
        if prediction_names is not None and names != prediction_names:
            raise ValueError("Prediction feature schema does not match the fitted model")
        if np.isinf(matrix).any():
            raise ValueError("Regime features may contain NaN gaps but not infinities")
        return matrix, names

    @staticmethod
    def _coerce_orientations(values: Sequence[int], dimension: int) -> tuple[int, ...]:
        orientations = tuple(values)
        if len(orientations) != dimension or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in (-1, 1)
            for value in orientations
        ):
            raise ValueError("stress_orientations must contain one -1/+1 integer per feature")
        return tuple(int(value) for value in orientations)

    @staticmethod
    def _initial_parameters(
        scaled: np.ndarray,
        complete: np.ndarray,
        slices: Sequence[slice],
        orientations: np.ndarray,
        config: RegimeHMMConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observations = scaled[complete]
        score = np.mean(observations * orientations[None, :], axis=1)
        order = np.argsort(score, kind="stable")
        assignment = np.zeros(len(observations), dtype=int)
        assignment[order[len(order) // 2 :]] = 1
        means = np.vstack([observations[assignment == state].mean(axis=0) for state in range(2)])
        variances = np.vstack(
            [
                np.maximum(
                    np.mean((observations[assignment == state] - means[state]) ** 2, axis=0),
                    config.variance_floor,
                )
                for state in range(2)
            ]
        )
        full_assignment = np.full(len(scaled), -1, dtype=int)
        full_assignment[np.flatnonzero(complete)] = assignment
        pi_counts = np.asarray(config.pi_additive, dtype=float).copy()
        transition_counts = np.asarray(config.transition_additive, dtype=float).copy()
        for sequence_slice in slices:
            states = full_assignment[sequence_slice]
            pi_counts[states[0]] += 1.0
            if len(states) > 1:
                np.add.at(transition_counts, (states[:-1], states[1:]), 1.0)
        pi = pi_counts / pi_counts.sum()
        transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        return _oriented_order(pi, transition, means, variances, orientations)

    @staticmethod
    def _expectation(
        scaled: np.ndarray,
        slices: Sequence[slice],
        pi: np.ndarray,
        transition: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> tuple[float, list[np.ndarray], np.ndarray, np.ndarray]:
        likelihood = 0.0
        gammas: list[np.ndarray] = []
        transition_counts = np.zeros((2, 2), dtype=float)
        pi_counts = np.zeros(2, dtype=float)
        for sequence_slice in slices:
            sequence = scaled[sequence_slice]
            sequence_likelihood, gamma, xi_sum = _forward_backward(
                sequence, pi, transition, means, variances
            )
            likelihood += sequence_likelihood
            gammas.append(gamma)
            transition_counts += xi_sum
            pi_counts += gamma[0]
        return float(likelihood), gammas, transition_counts, pi_counts

    @staticmethod
    def _maximization(
        scaled: np.ndarray,
        slices: Sequence[slice],
        gammas: Sequence[np.ndarray],
        transition_counts: np.ndarray,
        pi_counts: np.ndarray,
        orientations: np.ndarray,
        config: RegimeHMMConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        occupancy = np.sum([gamma.sum(axis=0) for gamma in gammas], axis=0)
        weighted_sum = np.zeros((2, scaled.shape[1]), dtype=float)
        for sequence_slice, gamma in zip(slices, gammas):
            weighted_sum += gamma.T @ scaled[sequence_slice]
        means = weighted_sum / occupancy[:, None]
        weighted_variance = np.zeros_like(means)
        for sequence_slice, gamma in zip(slices, gammas):
            observations = scaled[sequence_slice]
            differences = observations[:, None, :] - means[None, :, :]
            weighted_variance += np.sum(gamma[:, :, None] * differences * differences, axis=0)
        variances = np.maximum(weighted_variance / occupancy[:, None], config.variance_floor)
        pi_smoothed = pi_counts + np.asarray(config.pi_additive, dtype=float)
        pi = pi_smoothed / pi_smoothed.sum()
        transition_smoothed = transition_counts + np.asarray(config.transition_additive, dtype=float)
        transition = transition_smoothed / transition_smoothed.sum(axis=1, keepdims=True)
        pi, transition, means, variances = _oriented_order(
            pi, transition, means, variances, orientations
        )
        return pi, transition, means, variances, occupancy

    def _fit_hmm(
        self,
        scaled: np.ndarray,
        complete: np.ndarray,
        orientations: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        float,
        float,
    ]:
        slices = _sequence_slices(complete)
        pi, transition, means, variances = self._initial_parameters(
            scaled, complete, slices, orientations, self.config
        )
        previous_likelihood, _, _, _ = self._expectation(
            scaled, slices, pi, transition, means, variances
        )
        previous_objective = _regularized_log_objective(
            previous_likelihood, pi, transition, self.config
        )
        converged = False
        occupancy = np.zeros(2, dtype=float)
        completed_iterations = 0
        for iteration in range(1, self.config.max_iterations + 1):
            _, gammas, transition_counts, pi_counts = self._expectation(
                scaled, slices, pi, transition, means, variances
            )
            candidate = self._maximization(
                scaled,
                slices,
                gammas,
                transition_counts,
                pi_counts,
                orientations,
                self.config,
            )
            pi, transition, means, variances, occupancy = candidate
            likelihood, _, _, _ = self._expectation(
                scaled, slices, pi, transition, means, variances
            )
            objective = _regularized_log_objective(
                likelihood, pi, transition, self.config
            )
            if _likelihood_decreased_beyond_guard(previous_objective, objective):
                raise RuntimeError(
                    "HMM regularized objective decreased beyond the frozen numerical guard: "
                    f"previous={previous_objective:.17g}, current={objective:.17g}, "
                    f"iteration={iteration}"
                )
            relative_change = abs(objective - previous_objective) / max(
                1.0, abs(previous_objective)
            )
            completed_iterations = iteration
            if iteration >= self.config.min_iterations and relative_change <= self.config.relative_tolerance:
                converged = True
                previous_likelihood = likelihood
                previous_objective = objective
                break
            previous_likelihood = likelihood
            previous_objective = objective
        if not converged:
            raise RuntimeError("HMM Baum-Welch solver did not converge within 100 iterations")
        final_likelihood, final_gammas, _, _ = self._expectation(
            scaled, slices, pi, transition, means, variances
        )
        occupancy = np.sum([gamma.sum(axis=0) for gamma in final_gammas], axis=0)
        final_objective = _regularized_log_objective(
            final_likelihood, pi, transition, self.config
        )
        if (occupancy < self.config.min_effective_occupancy).any():
            raise ValueError("HMM state effective occupancy is below the frozen minimum")
        return (
            pi,
            transition,
            means,
            variances,
            occupancy,
            completed_iterations,
            final_likelihood,
            final_objective,
        )

    def fit(
        self,
        features: Any,
        cash_beats_long_10bps: Sequence[float],
        edge_10bps: Sequence[float],
        *,
        head_name: str,
        feature_names: Sequence[str] | None = None,
        stress_orientations: Sequence[int],
        fit_metadata: Mapping[str, Any] | RegimeFitMetadata,
    ) -> "RegimeHeadModel":
        self.config.validate()
        matrix, names = self._coerce_feature_matrix(features, feature_names)
        if not isinstance(head_name, str) or not head_name or head_name.strip() != head_name:
            raise ValueError("head_name must be a non-empty canonical string")
        orientations_tuple = self._coerce_orientations(stress_orientations, len(names))
        binary = np.asarray(cash_beats_long_10bps, dtype=float)
        edge = np.asarray(edge_10bps, dtype=float)
        if binary.shape != (len(matrix),) or edge.shape != (len(matrix),):
            raise ValueError("Regime outcome targets must align with the feature timeline")
        if np.isinf(binary).any() or np.isinf(edge).any():
            raise ValueError("Regime outcome targets may be NaN but not infinite")
        if not np.array_equal(np.isfinite(binary), np.isfinite(edge)):
            raise ValueError("Binary and edge target availability must match")
        if not np.isin(binary[np.isfinite(binary)], (0.0, 1.0)).all():
            raise ValueError("cash_beats_long_10bps must be binary zero/one where available")
        metadata = RegimeFitMetadata.coerce(fit_metadata)
        complete = np.isfinite(matrix).all(axis=1)
        complete_count = int(complete.sum())
        if complete_count < self.config.min_complete_emissions:
            raise ValueError(
                f"HMM requires at least {self.config.min_complete_emissions} complete emissions"
            )
        observations = matrix[complete]
        medians = np.median(observations, axis=0)
        mads = np.median(np.abs(observations - medians), axis=0)
        scales = np.maximum(
            self.config.raw_mad_multiplier * mads, self.config.raw_scale_floor
        )
        scaled = np.full_like(matrix, np.nan, dtype=float)
        scaled[complete] = np.clip(
            (observations - medians) / scales,
            -self.config.raw_z_clip,
            self.config.raw_z_clip,
        )
        fitted = self._fit_hmm(
            scaled, complete, np.asarray(orientations_tuple, dtype=float)
        )
        (
            self.initial_probabilities,
            self.transition_matrix,
            self.means,
            self.variances,
            self.effective_occupancy,
            self.iterations,
            self.final_log_likelihood,
            self.final_regularized_objective,
        ) = fitted
        self.head_name = head_name
        self.feature_names = names
        self.stress_orientations = orientations_tuple
        self.fit_metadata = metadata
        self.raw_medians = medians
        self.raw_scales = scales
        self.training_row_count = int(len(matrix))
        self.complete_emission_count = complete_count
        self.sequence_count = len(_sequence_slices(complete))
        self.converged = True
        filtered = self._filter_scaled(scaled, previous_posterior=None)
        self.training_terminal_contiguous = bool(complete[-1])
        self.training_terminal_state_probability = (
            filtered[-1].copy() if complete[-1] else self.initial_probabilities.copy()
        )
        self.outcome_head = RegimeOutcomeHead(self.config).fit(filtered, binary, edge)
        self._validate_fitted_semantics()
        return self

    def _scaled_prediction_matrix(self, matrix: np.ndarray) -> np.ndarray:
        complete = np.isfinite(matrix).all(axis=1)
        scaled = np.full_like(matrix, np.nan, dtype=float)
        scaled[complete] = np.clip(
            (matrix[complete] - self.raw_medians) / self.raw_scales,
            -self.config.raw_z_clip,
            self.config.raw_z_clip,
        )
        return scaled

    def _filter_scaled(
        self,
        scaled: np.ndarray,
        *,
        previous_posterior: np.ndarray | None,
    ) -> np.ndarray:
        result = np.full((len(scaled), 2), np.nan, dtype=float)
        active = previous_posterior is not None
        previous = None if previous_posterior is None else previous_posterior.copy()
        for index, observation in enumerate(scaled):
            if not np.isfinite(observation).all():
                active = False
                previous = None
                continue
            prior = previous @ self.transition_matrix if active else self.initial_probabilities
            log_weight = np.log(prior) + _emission_log_probability(
                observation.reshape(1, -1), self.means, self.variances
            )[0]
            posterior = np.exp(log_weight - float(_logsumexp(log_weight)))
            posterior /= posterior.sum()
            result[index] = posterior
            previous = posterior
            active = True
        return result

    def filter_state_probabilities(
        self,
        features: Any,
        *,
        initial_state_probability: Sequence[float] | None = None,
        continue_from_training: bool = True,
    ) -> np.ndarray:
        self._require_fitted()
        matrix, _ = self._coerce_feature_matrix(
            features, None, prediction_names=self.feature_names
        )
        if not isinstance(continue_from_training, bool):
            raise ValueError("continue_from_training must be boolean")
        previous: np.ndarray | None
        if initial_state_probability is not None:
            previous = np.asarray(initial_state_probability, dtype=float)
            if previous.shape != (2,) or not np.isfinite(previous).all() or (previous < 0).any() or not math.isclose(float(previous.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("initial_state_probability must be a normalized length-two vector")
        elif continue_from_training and self.training_terminal_contiguous:
            previous = self.training_terminal_state_probability
        else:
            previous = None
        return self._filter_scaled(
            self._scaled_prediction_matrix(matrix), previous_posterior=previous
        )

    predict_state_probabilities = filter_state_probabilities

    def predict_components(
        self,
        features: Any,
        *,
        initial_state_probability: Sequence[float] | None = None,
        continue_from_training: bool = True,
    ) -> dict[str, np.ndarray]:
        state = self.filter_state_probabilities(
            features,
            initial_state_probability=initial_state_probability,
            continue_from_training=continue_from_training,
        )
        outcomes = self.outcome_head.predict_components(state)
        available = np.isfinite(state).all(axis=1)
        return {
            "state_probabilities": state,
            "stress_probability": state[:, 1].copy(),
            **outcomes,
            "observation_available": available,
        }

    def predict_proba(self, features: Any, **kwargs: Any) -> np.ndarray:
        return self.predict_components(features, **kwargs)["cash_beats_long_probability"]

    def predict_cash_probability(self, features: Any, **kwargs: Any) -> np.ndarray:
        return self.predict_proba(features, **kwargs)

    def predict_expected_edge(self, features: Any, **kwargs: Any) -> np.ndarray:
        return self.predict_components(features, **kwargs)["expected_edge_10bps"]

    def _parameter_payload(self) -> dict[str, Any]:
        return {
            "preprocessing": {
                "raw_medians": _encode_vector(self.raw_medians),
                "raw_scales": _encode_vector(self.raw_scales),
            },
            "hmm": {
                "initial_probabilities": _encode_vector(self.initial_probabilities),
                "transition_matrix": _encode_matrix(self.transition_matrix),
                "means": _encode_matrix(self.means),
                "variances": _encode_matrix(self.variances),
                "training_terminal_state_probability": _encode_vector(
                    self.training_terminal_state_probability
                ),
            },
            "outcome_head": self.outcome_head._payload(),
        }

    def _payload(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.fit_metadata is not None
        parameters = self._parameter_payload()
        return {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "config": _config_state(self.config),
            "head_name": self.head_name,
            "feature_names": list(self.feature_names),
            "stress_orientations": list(self.stress_orientations),
            "fit_metadata": asdict(self.fit_metadata),
            "training": {
                "row_count": self.training_row_count,
                "complete_emission_count": self.complete_emission_count,
                "sequence_count": self.sequence_count,
                "effective_occupancy": _encode_vector(self.effective_occupancy),
                "iterations": self.iterations,
                "converged": self.converged,
                "final_log_likelihood": _float_hex(self.final_log_likelihood),
                "final_regularized_objective": _float_hex(
                    self.final_regularized_objective
                ),
                "terminal_contiguous": self.training_terminal_contiguous,
            },
            "parameters": parameters,
            "parameters_sha256": _sha256_json(parameters),
        }

    def to_state(self) -> dict[str, Any]:
        payload = self._payload()
        return json.loads(
            _canonical_json({"payload": payload, "payload_sha256": _sha256_json(payload)})
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
    def load(cls, path: str | os.PathLike[str]) -> "RegimeHeadModel":
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read a valid regime head model state") from exc
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "RegimeHeadModel":
        if not isinstance(state, Mapping):
            raise ValueError("Regime head model state must be an object")
        _expect_keys(state, {"payload", "payload_sha256"}, "state")
        payload = state["payload"]
        if not isinstance(payload, Mapping):
            raise ValueError("Regime head model payload must be an object")
        payload_hash = state["payload_sha256"]
        if not _is_sha256(payload_hash):
            raise ValueError("Regime head model payload hash is invalid")
        if not hmac.compare_digest(_sha256_json(payload), payload_hash):
            raise ValueError("Regime head model state checksum mismatch")
        _expect_keys(
            payload,
            {
                "state_schema_version", "model_type", "config", "head_name",
                "feature_names", "stress_orientations", "fit_metadata", "training",
                "parameters", "parameters_sha256",
            },
            "payload",
        )
        if (
            isinstance(payload["state_schema_version"], bool)
            or not isinstance(payload["state_schema_version"], int)
            or payload["state_schema_version"] != STATE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported regime head model state schema")
        if payload["model_type"] != MODEL_TYPE:
            raise ValueError("Unexpected regime head model type")
        config = _config_from_state(payload["config"])
        head_name = payload["head_name"]
        if not isinstance(head_name, str) or not head_name or head_name.strip() != head_name:
            raise ValueError("Regime head name is invalid")
        raw_names = payload["feature_names"]
        if (
            not isinstance(raw_names, list)
            or not raw_names
            or any(not isinstance(name, str) or not name for name in raw_names)
            or len(set(raw_names)) != len(raw_names)
        ):
            raise ValueError("Regime feature schema is invalid")
        names = tuple(raw_names)
        orientations = cls._coerce_orientations(payload["stress_orientations"], len(names))
        metadata = RegimeFitMetadata.coerce(payload["fit_metadata"])
        training = payload["training"]
        if not isinstance(training, Mapping):
            raise ValueError("Regime training metadata must be an object")
        _expect_keys(
            training,
            {
                "row_count", "complete_emission_count", "sequence_count",
                "effective_occupancy", "iterations", "converged",
                "final_log_likelihood", "final_regularized_objective",
                "terminal_contiguous",
            },
            "training",
        )
        row_count = _strict_int(training["row_count"], "row_count", minimum=1)
        complete_count = _strict_int(
            training["complete_emission_count"],
            "complete_emission_count",
            minimum=config.min_complete_emissions,
        )
        sequence_count = _strict_int(training["sequence_count"], "sequence_count", minimum=1)
        occupancy = _decode_vector(training["effective_occupancy"], "effective_occupancy", 2)
        iterations = _strict_int(training["iterations"], "iterations", minimum=config.min_iterations)
        converged = training["converged"]
        if converged is not True:
            raise ValueError("Serialized regime HMM must be converged")
        final_likelihood = _decode_float_hex(
            training["final_log_likelihood"], "final_log_likelihood"
        )
        final_objective = _decode_float_hex(
            training["final_regularized_objective"],
            "final_regularized_objective",
        )
        terminal_contiguous = training["terminal_contiguous"]
        if not isinstance(terminal_contiguous, bool):
            raise ValueError("terminal_contiguous must be boolean")
        parameters = payload["parameters"]
        if not isinstance(parameters, Mapping):
            raise ValueError("Regime model parameters must be an object")
        parameter_hash = payload["parameters_sha256"]
        if not _is_sha256(parameter_hash) or not hmac.compare_digest(
            _sha256_json(parameters), parameter_hash
        ):
            raise ValueError("Regime model parameter checksum mismatch")
        _expect_keys(parameters, {"preprocessing", "hmm", "outcome_head"}, "parameters")
        preprocessing = parameters["preprocessing"]
        if not isinstance(preprocessing, Mapping):
            raise ValueError("Regime preprocessing must be an object")
        _expect_keys(preprocessing, {"raw_medians", "raw_scales"}, "preprocessing")
        medians = _decode_vector(preprocessing["raw_medians"], "raw_medians", len(names))
        scales = _decode_vector(preprocessing["raw_scales"], "raw_scales", len(names))
        hmm = parameters["hmm"]
        if not isinstance(hmm, Mapping):
            raise ValueError("Regime HMM parameters must be an object")
        _expect_keys(
            hmm,
            {
                "initial_probabilities", "transition_matrix", "means", "variances",
                "training_terminal_state_probability",
            },
            "hmm parameters",
        )
        pi = _decode_vector(hmm["initial_probabilities"], "initial_probabilities", 2)
        transition = _decode_matrix(hmm["transition_matrix"], "transition_matrix", (2, 2))
        means = _decode_matrix(hmm["means"], "means", (2, len(names)))
        variances = _decode_matrix(hmm["variances"], "variances", (2, len(names)))
        terminal = _decode_vector(
            hmm["training_terminal_state_probability"],
            "training_terminal_state_probability",
            2,
        )
        outcome_payload = parameters["outcome_head"]
        if not isinstance(outcome_payload, Mapping):
            raise ValueError("Regime outcome payload must be an object")
        outcome = RegimeOutcomeHead(config)
        outcome._load_payload(outcome_payload)
        model = cls(config)
        model.head_name = head_name
        model.feature_names = names
        model.stress_orientations = orientations
        model.fit_metadata = metadata
        model.raw_medians = medians
        model.raw_scales = scales
        model.initial_probabilities = pi
        model.transition_matrix = transition
        model.means = means
        model.variances = variances
        model.effective_occupancy = occupancy
        model.training_row_count = row_count
        model.complete_emission_count = complete_count
        model.sequence_count = sequence_count
        model.iterations = iterations
        model.converged = converged
        model.final_log_likelihood = final_likelihood
        model.final_regularized_objective = final_objective
        model.training_terminal_state_probability = terminal
        model.training_terminal_contiguous = terminal_contiguous
        model.outcome_head = outcome
        model._validate_fitted_semantics()
        if model.model_sha256 != payload_hash:
            raise ValueError("Decoded regime head model hash changed unexpectedly")
        return model

    def _validate_fitted_semantics(self) -> None:
        self.config.validate()
        if self.fit_metadata is None:
            raise ValueError("Fitted regime model is missing fit metadata")
        self.fit_metadata.validate()
        if not self.head_name or self.head_name.strip() != self.head_name:
            raise ValueError("Fitted regime model head name is invalid")
        dimension = len(self.feature_names)
        if dimension < 1 or len(set(self.feature_names)) != dimension:
            raise ValueError("Fitted regime feature schema is invalid")
        if self._coerce_orientations(self.stress_orientations, dimension) != self.stress_orientations:
            raise ValueError("Fitted regime stress orientations are invalid")
        if self.raw_medians.shape != (dimension,) or self.raw_scales.shape != (dimension,):
            raise ValueError("Fitted regime preprocessing shape is invalid")
        if not np.isfinite(self.raw_medians).all() or not np.isfinite(self.raw_scales).all():
            raise ValueError("Fitted regime preprocessing is non-finite")
        if (self.raw_scales < self.config.raw_scale_floor).any():
            raise ValueError("Fitted regime raw scale violates the frozen floor")
        if self.initial_probabilities.shape != (2,) or self.transition_matrix.shape != (2, 2):
            raise ValueError("Fitted regime Markov parameter shape is invalid")
        if self.means.shape != (2, dimension) or self.variances.shape != (2, dimension):
            raise ValueError("Fitted regime emission parameter shape is invalid")
        for values in (
            self.initial_probabilities,
            self.transition_matrix,
            self.means,
            self.variances,
            self.effective_occupancy,
            self.training_terminal_state_probability,
        ):
            if not np.isfinite(values).all():
                raise ValueError("Fitted regime parameters contain non-finite values")
        if (self.initial_probabilities <= 0.0).any() or not math.isclose(
            float(self.initial_probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Fitted regime initial probabilities are invalid")
        if (self.transition_matrix <= 0.0).any() or not np.allclose(
            self.transition_matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("Fitted regime transition matrix is invalid")
        if (self.variances < self.config.variance_floor).any():
            raise ValueError("Fitted regime variance violates the frozen floor")
        oriented_stress = np.mean(
            self.means * np.asarray(self.stress_orientations)[None, :], axis=1
        )
        if oriented_stress[0] > oriented_stress[1]:
            raise ValueError("Fitted regime state order is not calm then stress")
        if oriented_stress[0] == oriented_stress[1] and tuple(self.means[0]) > tuple(self.means[1]):
            raise ValueError("Fitted regime state tie order is not lexicographic")
        if self.effective_occupancy.shape != (2,) or (
            self.effective_occupancy < self.config.min_effective_occupancy
        ).any():
            raise ValueError("Fitted regime occupancy violates the frozen minimum")
        if self.complete_emission_count < self.config.min_complete_emissions:
            raise ValueError("Fitted regime model has too few complete emissions")
        if not math.isclose(
            float(self.effective_occupancy.sum()),
            self.complete_emission_count,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError("Fitted regime occupancy does not match complete emissions")
        if not self.complete_emission_count <= self.training_row_count or self.sequence_count < 1:
            raise ValueError("Fitted regime training counts are inconsistent")
        if not self.converged or not self.config.min_iterations <= self.iterations <= self.config.max_iterations:
            raise ValueError("Fitted regime solver convergence metadata is invalid")
        if not math.isfinite(self.final_log_likelihood):
            raise ValueError("Fitted regime likelihood is non-finite")
        if not math.isfinite(self.final_regularized_objective):
            raise ValueError("Fitted regime regularized objective is non-finite")
        expected_objective = _regularized_log_objective(
            self.final_log_likelihood,
            self.initial_probabilities,
            self.transition_matrix,
            self.config,
        )
        if not math.isclose(
            self.final_regularized_objective,
            expected_objective,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Fitted regime regularized objective is inconsistent")
        if self.training_terminal_state_probability.shape != (2,) or (
            self.training_terminal_state_probability < 0.0
        ).any() or not math.isclose(
            float(self.training_terminal_state_probability.sum()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Fitted regime terminal state probability is invalid")
        if not self.training_terminal_contiguous and not np.array_equal(
            self.training_terminal_state_probability, self.initial_probabilities
        ):
            raise ValueError("A missing terminal emission must persist the reset distribution")
        self.outcome_head._validate_fitted_semantics()
        if self.outcome_head.sample_count > self.training_row_count:
            raise ValueError("Outcome sample count exceeds the training timeline")


RegimeHMM = RegimeHeadModel


__all__ = [
    "MODEL_TYPE",
    "STATE_SCHEMA_VERSION",
    "RegimeFitMetadata",
    "RegimeHeadModel",
    "RegimeHMM",
    "RegimeHMMConfig",
    "RegimeOutcomeHead",
]
