"""Causal walk-forward fitting for the frozen sector-breadth experiment.

The module is intentionally data-source agnostic.  It accepts the
point-in-time feature/label frame produced by :mod:`sector_breadth_features`
and has no loader, network, or performance-selection entry point.

Development is limited to the seven already-preregistered 2005--2018
expanding, purged folds.  A separately serialized policy can then be fitted
through the 2018 label boundary and used, without refitting, for 2019--2023.
Only that intermediate policy can authorize the final 2023-boundary refit;
this module deliberately has no 2024+ prediction API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from .direct_edge_features import DIRECT_EDGE_LABEL_COLUMNS, PRICE_FEATURE_COLUMNS
from .direct_edge_gam import DirectEdgeGAM
from .direct_edge_walkforward import (
    DIRECT_EDGE_CANDIDATES,
    WALK_FORWARD_FOLDS,
    DirectEdgeCandidate,
    WalkForwardFold,
    five_session_cash_policy,
)
from .sector_breadth_features import SECTOR_BREADTH_FEATURE_COLUMNS


DEVELOPMENT_FIRST_DECISION: Final[pd.Timestamp] = pd.Timestamp("2005-01-01")
DEVELOPMENT_LAST_DECISION: Final[pd.Timestamp] = pd.Timestamp("2018-12-31")
INTERMEDIATE_FIRST_DECISION: Final[pd.Timestamp] = pd.Timestamp("2019-01-01")
INTERMEDIATE_LAST_DECISION: Final[pd.Timestamp] = pd.Timestamp("2023-12-31")
INTERMEDIATE_TRAINING_BOUNDARY: Final[pd.Timestamp] = pd.Timestamp("2019-01-01")
FINAL_TRAINING_BOUNDARY: Final[pd.Timestamp] = pd.Timestamp("2024-01-01")
MAXIMUM_ACCEPTED_DECISION: Final[pd.Timestamp] = INTERMEDIATE_LAST_DECISION

SECTOR_BREADTH_MODEL_FAMILIES: Final[tuple[str, str]] = (
    "price_only",
    "sector_breadth",
)
MODEL_FAMILIES = SECTOR_BREADTH_MODEL_FAMILIES
SECTOR_BREADTH_WALK_FORWARD_FOLDS: Final[tuple[WalkForwardFold, ...]] = (
    WALK_FORWARD_FOLDS
)
POLICY_STATE_SCHEMA_VERSION: Final[int] = 1
POLICY_TYPE: Final[str] = "aapl_sector_breadth_frozen_policy_v1"
AUTHORIZED_TRAINING_BOUNDARIES: Final[frozenset[pd.Timestamp]] = frozenset(
    {
        *(pd.Timestamp(f"{fold.first_year}-01-01") for fold in WALK_FORWARD_FOLDS),
        INTERMEDIATE_TRAINING_BOUNDARY,
        FINAL_TRAINING_BOUNDARY,
    }
)

_READINESS_COLUMNS: Final[tuple[str, ...]] = (
    "price_features_ready",
    "sector_breadth_features_ready",
    "sector_breadth_price_only_fallback",
    "label_available",
)
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *PRICE_FEATURE_COLUMNS,
        *SECTOR_BREADTH_FEATURE_COLUMNS,
        *DIRECT_EDGE_LABEL_COLUMNS,
        *_READINESS_COLUMNS,
        "sector_breadth_status",
    }
)
_VALID_SECTOR_STATUSES: Final[frozenset[str]] = frozenset(
    {"ready", "insufficient_history", "missing_context"}
)
_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


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


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _candidate_from_value(value: DirectEdgeCandidate | str) -> DirectEdgeCandidate:
    if isinstance(value, DirectEdgeCandidate):
        candidate = value
    elif isinstance(value, str):
        candidate_id = value.strip()
        for family in SECTOR_BREADTH_MODEL_FAMILIES:
            prefix = f"{family}_"
            if candidate_id.startswith(prefix):
                candidate_id = candidate_id[len(prefix) :]
                break
        matches = [
            item for item in DIRECT_EDGE_CANDIDATES if item.candidate_id == candidate_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Unknown frozen direct-edge candidate: {value!r}")
        candidate = matches[0]
    else:
        raise TypeError("selected_candidate must be a DirectEdgeCandidate or string")
    if candidate not in DIRECT_EDGE_CANDIDATES:
        raise ValueError("Candidate is not in the frozen four-gate grid")
    return candidate


def _feature_names(model_family: str) -> tuple[str, ...]:
    if model_family == "price_only":
        return tuple(PRICE_FEATURE_COLUMNS)
    if model_family == "sector_breadth":
        return tuple(PRICE_FEATURE_COLUMNS) + tuple(SECTOR_BREADTH_FEATURE_COLUMNS)
    raise ValueError(f"Unknown sector-breadth model family: {model_family!r}")


def sector_breadth_candidate_cash_target_column(
    model_family: str,
    candidate_id: str,
) -> str:
    _feature_names(model_family)
    candidate = _candidate_from_value(candidate_id)
    return f"cash_target_{model_family}_{candidate.candidate_id}"


def sector_breadth_candidate_cash_block_start_column(
    model_family: str,
    candidate_id: str,
) -> str:
    _feature_names(model_family)
    candidate = _candidate_from_value(candidate_id)
    return f"cash_block_start_{model_family}_{candidate.candidate_id}"


@dataclass(frozen=True)
class FrozenSectorBreadthPolicy:
    """One fitted model plus one preregistered gate and its audit metadata."""

    model_family: str
    candidate: DirectEdgeCandidate
    training_boundary: pd.Timestamp
    feature_names: tuple[str, ...]
    model: DirectEdgeGAM
    model_sha256: str
    training_sha256: str
    training_count: int
    training_positive_count: int
    training_max_decision_date: pd.Timestamp
    training_max_label_maturity_date: pd.Timestamp
    baseline_cash_win_probability: float
    baseline_mean_net_edge: float

    def __post_init__(self) -> None:
        expected_names = _feature_names(self.model_family)
        if tuple(self.feature_names) != expected_names:
            raise ValueError("Frozen policy feature schema does not match its family")
        if self.candidate not in DIRECT_EDGE_CANDIDATES:
            raise ValueError("Frozen policy candidate is outside the fixed gate grid")
        boundary = _normalize_timestamp(
            self.training_boundary, field_name="training_boundary"
        )
        if boundary not in AUTHORIZED_TRAINING_BOUNDARIES:
            raise ValueError("Frozen policy uses an unauthorized training boundary")
        object.__setattr__(self, "training_boundary", boundary)
        maximum_decision = _normalize_timestamp(
            self.training_max_decision_date,
            field_name="training_max_decision_date",
        )
        maximum_maturity = _normalize_timestamp(
            self.training_max_label_maturity_date,
            field_name="training_max_label_maturity_date",
        )
        object.__setattr__(self, "training_max_decision_date", maximum_decision)
        object.__setattr__(
            self, "training_max_label_maturity_date", maximum_maturity
        )
        if maximum_decision >= boundary or maximum_maturity >= boundary:
            raise ValueError(
                "Frozen policy contains a post-boundary or future-mature label"
            )
        if (
            isinstance(self.training_count, bool)
            or not isinstance(self.training_count, (int, np.integer))
            or int(self.training_count) < 2
        ):
            raise ValueError("Frozen policy training_count must be an integer >= 2")
        if (
            isinstance(self.training_positive_count, bool)
            or not isinstance(self.training_positive_count, (int, np.integer))
            or not 0 < int(self.training_positive_count) < int(self.training_count)
        ):
            raise ValueError("Frozen policy training data must contain both classes")
        if not _is_sha256(self.model_sha256) or not _is_sha256(
            self.training_sha256
        ):
            raise ValueError("Frozen policy contains an invalid SHA-256")
        observed_model_hash = getattr(self.model, "model_sha256", None)
        if observed_model_hash != self.model_sha256:
            raise ValueError("Frozen policy model hash does not match its model")
        observed_names = tuple(getattr(self.model, "feature_names", ()))
        if observed_names and observed_names != expected_names:
            raise ValueError("Frozen policy model uses a different feature schema")
        probability = float(self.baseline_cash_win_probability)
        edge = float(self.baseline_mean_net_edge)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Frozen policy baseline probability is invalid")
        if not math.isfinite(edge):
            raise ValueError("Frozen policy baseline edge is invalid")

    @property
    def candidate_id(self) -> str:
        return f"{self.model_family}_{self.candidate.candidate_id}"

    def _payload(self) -> dict[str, Any]:
        to_state = getattr(self.model, "to_state", None)
        if not callable(to_state):
            raise ValueError("Frozen policy model cannot be serialized")
        return {
            "state_schema_version": POLICY_STATE_SCHEMA_VERSION,
            "policy_type": POLICY_TYPE,
            "model_family": self.model_family,
            "candidate": {
                "candidate_id": self.candidate.candidate_id,
                "probability_gate": float(self.candidate.probability_gate).hex(),
                "expected_edge_gate": float(
                    self.candidate.expected_edge_gate
                ).hex(),
            },
            "training_boundary": self.training_boundary.date().isoformat(),
            "feature_names": list(self.feature_names),
            "model_sha256": self.model_sha256,
            "training_sha256": self.training_sha256,
            "training": {
                "count": int(self.training_count),
                "positive_count": int(self.training_positive_count),
                "max_decision_date": self.training_max_decision_date.date().isoformat(),
                "max_label_maturity_date": (
                    self.training_max_label_maturity_date.date().isoformat()
                ),
                "baseline_cash_win_probability": float(
                    self.baseline_cash_win_probability
                ).hex(),
                "baseline_mean_net_edge": float(
                    self.baseline_mean_net_edge
                ).hex(),
            },
            "model_state": to_state(),
        }

    def to_state(self) -> dict[str, Any]:
        payload = self._payload()
        return {
            "payload": payload,
            "payload_sha256": _sha256_json(payload),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_state())

    def save(self, path: str | os.PathLike[str]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes((self.canonical_json() + "\n").encode("utf-8"))
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "FrozenSectorBreadthPolicy":
        try:
            state = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read a frozen sector-breadth policy") from exc
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FrozenSectorBreadthPolicy":
        if not isinstance(state, Mapping) or set(state) != {
            "payload",
            "payload_sha256",
        }:
            raise ValueError("Frozen policy state has invalid top-level keys")
        payload = state["payload"]
        digest = state["payload_sha256"]
        if (
            not isinstance(payload, Mapping)
            or not _is_sha256(digest)
            or not hmac.compare_digest(_sha256_json(payload), str(digest))
        ):
            raise ValueError("Frozen policy state checksum mismatch")
        expected_keys = {
            "state_schema_version",
            "policy_type",
            "model_family",
            "candidate",
            "training_boundary",
            "feature_names",
            "model_sha256",
            "training_sha256",
            "training",
            "model_state",
        }
        if set(payload) != expected_keys:
            raise ValueError("Frozen policy payload keys are invalid")
        if payload["state_schema_version"] != POLICY_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported frozen policy state schema")
        if payload["policy_type"] != POLICY_TYPE:
            raise ValueError("Unexpected frozen policy type")
        candidate_state = payload["candidate"]
        if not isinstance(candidate_state, Mapping) or set(candidate_state) != {
            "candidate_id",
            "probability_gate",
            "expected_edge_gate",
        }:
            raise ValueError("Frozen policy candidate state is invalid")
        candidate = _candidate_from_value(candidate_state["candidate_id"])
        if (
            candidate_state["probability_gate"]
            != float(candidate.probability_gate).hex()
            or candidate_state["expected_edge_gate"]
            != float(candidate.expected_edge_gate).hex()
        ):
            raise ValueError("Frozen policy gate values changed")
        training = payload["training"]
        if not isinstance(training, Mapping) or set(training) != {
            "count",
            "positive_count",
            "max_decision_date",
            "max_label_maturity_date",
            "baseline_cash_win_probability",
            "baseline_mean_net_edge",
        }:
            raise ValueError("Frozen policy training metadata is invalid")
        model = DirectEdgeGAM.from_state(payload["model_state"])
        try:
            probability = float.fromhex(training["baseline_cash_win_probability"])
            edge = float.fromhex(training["baseline_mean_net_edge"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Frozen policy baselines are invalid") from exc
        return cls(
            model_family=payload["model_family"],
            candidate=candidate,
            training_boundary=pd.Timestamp(payload["training_boundary"]),
            feature_names=tuple(payload["feature_names"]),
            model=model,
            model_sha256=payload["model_sha256"],
            training_sha256=payload["training_sha256"],
            training_count=training["count"],
            training_positive_count=training["positive_count"],
            training_max_decision_date=pd.Timestamp(training["max_decision_date"]),
            training_max_label_maturity_date=pd.Timestamp(
                training["max_label_maturity_date"]
            ),
            baseline_cash_win_probability=probability,
            baseline_mean_net_edge=edge,
        )


def _normalize_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} cannot be missing")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _canonical_frame(feature_label_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_label_frame, pd.DataFrame):
        raise TypeError("feature_label_frame must be a pandas DataFrame")
    if feature_label_frame.empty:
        raise ValueError("feature_label_frame cannot be empty")
    missing = sorted(_REQUIRED_COLUMNS.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(feature_label_frame.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("feature_label_frame index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise ValueError("feature_label_frame contains duplicate decision dates")
    if not index.is_monotonic_increasing:
        raise ValueError("feature_label_frame must be in chronological order")
    forbidden = index > MAXIMUM_ACCEPTED_DECISION
    if bool(forbidden.any()):
        first = index[forbidden][0]
        raise ValueError(
            "Sector-breadth fitting refuses decisions after 2023-12-31; "
            f"first forbidden date is {first.date().isoformat()}"
        )

    result = feature_label_frame.copy()
    result.index = index
    result.index.name = "decision_date"
    for name in _READINESS_COLUMNS:
        values = result[name].to_numpy(dtype=object)
        if any(not isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(f"{name} must contain non-missing booleans")
        result[name] = result[name].astype(bool)
    statuses = result["sector_breadth_status"]
    if statuses.isna().any() or not statuses.astype(str).isin(_VALID_SECTOR_STATUSES).all():
        raise ValueError("sector_breadth_status contains an invalid value")
    if bool(
        (
            result["sector_breadth_features_ready"]
            & result["sector_breadth_price_only_fallback"]
        ).any()
    ):
        raise ValueError("A ready sector-breadth row cannot request price fallback")

    raw_maturity = result["label_maturity_date"]
    maturity = pd.to_datetime(raw_maturity, errors="coerce")
    malformed_maturity = raw_maturity.notna() & maturity.isna()
    if bool(malformed_maturity.any()):
        raise ValueError("label_maturity_date contains an invalid date")
    if isinstance(maturity.dtype, pd.DatetimeTZDtype):
        maturity = maturity.dt.tz_localize(None)
    maturity = maturity.dt.normalize()
    result["label_maturity_date"] = maturity

    raw_binary = result["cash_beats_long_10bps"]
    raw_edge = result["cash_active_log_edge_10bps"]
    binary = pd.to_numeric(raw_binary, errors="coerce")
    edge = pd.to_numeric(raw_edge, errors="coerce")
    if bool((raw_binary.notna() & binary.isna()).any()) or bool(
        (raw_edge.notna() & edge.isna()).any()
    ):
        raise ValueError("Direct-edge labels must be numeric")
    computed_available = maturity.notna() & binary.notna() & edge.notna()
    if not np.array_equal(
        computed_available.to_numpy(dtype=bool),
        result["label_available"].to_numpy(dtype=bool),
    ):
        raise ValueError("label_available does not match the direct-edge labels")
    future_mature = computed_available & (maturity > INTERMEDIATE_LAST_DECISION)
    if bool(future_mature.any()):
        first = result.index[future_mature.to_numpy(dtype=bool)][0]
        raise ValueError(
            "Development and intermediate validation refuse labels maturing "
            "after 2023-12-31; first forbidden decision date is "
            f"{first.date().isoformat()}"
        )
    if bool(computed_available.any()):
        available_binary = binary.loc[computed_available].to_numpy(dtype=float)
        available_edge = edge.loc[computed_available].to_numpy(dtype=float)
        if (
            not np.isfinite(available_binary).all()
            or not np.isin(available_binary, (0.0, 1.0)).all()
        ):
            raise ValueError("cash_beats_long_10bps must be binary")
        if not np.isfinite(available_edge).all():
            raise ValueError("cash_active_log_edge_10bps must be finite")
        decision_values = result.index.to_numpy(dtype="datetime64[ns]")
        maturity_values = maturity.to_numpy(dtype="datetime64[ns]")
        bad_order = computed_available.to_numpy(dtype=bool) & (
            maturity_values <= decision_values
        )
        if bool(bad_order.any()):
            raise ValueError("A label must mature strictly after its decision")
    result["cash_beats_long_10bps"] = binary
    result["cash_active_log_edge_10bps"] = edge
    return result


def _finite_rows(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sector-breadth model inputs must be numeric") from exc
    return np.isfinite(values).all(axis=1)


def _ready_rows(frame: pd.DataFrame, model_family: str) -> np.ndarray:
    names = _feature_names(model_family)
    ready = frame["price_features_ready"].to_numpy(dtype=bool) & _finite_rows(
        frame, PRICE_FEATURE_COLUMNS
    )
    if model_family == "sector_breadth":
        ready &= frame["sector_breadth_features_ready"].to_numpy(dtype=bool)
        ready &= ~frame["sector_breadth_price_only_fallback"].to_numpy(dtype=bool)
        ready &= _finite_rows(frame, SECTOR_BREADTH_FEATURE_COLUMNS)
    if len(names) == 0:  # pragma: no cover - frozen schemas are nonempty
        raise RuntimeError("Frozen feature schema is empty")
    return ready


def _training_rows(
    frame: pd.DataFrame,
    *,
    model_family: str,
    boundary: pd.Timestamp,
) -> pd.DataFrame:
    boundary = _normalize_timestamp(boundary, field_name="training boundary")
    maturity = pd.to_datetime(frame["label_maturity_date"], errors="raise")
    # A feature ablation is meaningful only when both model families see the
    # same training cases.  The common support is the stricter breadth-ready
    # set; prediction readiness remains family-specific.
    common_training_ready = _ready_rows(frame, "sector_breadth")
    eligible = (
        (frame.index < boundary)
        & frame["label_available"].to_numpy(dtype=bool)
        & (maturity.to_numpy(dtype="datetime64[ns]") < boundary.to_datetime64())
        & common_training_ready
    )
    training = frame.loc[eligible].copy()
    if training.empty:
        raise ValueError(
            f"No eligible {model_family} training rows before {boundary.date()}"
        )
    maximum_maturity = pd.to_datetime(
        training["label_maturity_date"], errors="raise"
    ).max()
    if training.index.max() >= boundary or maximum_maturity >= boundary:
        raise RuntimeError("Future-mature labels entered a purged training set")
    return training


def _sha256_lines(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _fit_metadata(training: pd.DataFrame) -> dict[str, str]:
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    binary = training["cash_beats_long_10bps"].to_numpy(dtype=float)
    edge = training["cash_active_log_edge_10bps"].to_numpy(dtype=float)
    return {
        "training_start_date": training.index.min().date().isoformat(),
        "training_end_date": training.index.max().date().isoformat(),
        "maximum_label_maturity_date": maturity.max().date().isoformat(),
        "training_dates_sha256": _sha256_lines(
            [timestamp.date().isoformat() for timestamp in training.index]
        ),
        "binary_target_sha256": _sha256_lines(
            [float(value).hex() for value in binary]
        ),
        "edge_target_sha256": _sha256_lines(
            [float(value).hex() for value in edge]
        ),
    }


def _training_hash(
    training: pd.DataFrame,
    *,
    model_family: str,
    boundary: pd.Timestamp,
) -> str:
    names = _feature_names(model_family)
    feature_values = training.loc[:, list(names)].to_numpy(dtype=float)
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    binary = training["cash_beats_long_10bps"].to_numpy(dtype=float)
    edge = training["cash_active_log_edge_10bps"].to_numpy(dtype=float)
    payload = {
        "model_family": model_family,
        "training_boundary": boundary.date().isoformat(),
        "feature_names": list(names),
        "rows": [
            {
                "decision_date": decision.date().isoformat(),
                "label_maturity_date": maturity_value.date().isoformat(),
                "features": [float(value).hex() for value in row],
                "cash_beats_long_10bps": float(binary_value).hex(),
                "cash_active_log_edge_10bps": float(edge_value).hex(),
            }
            for decision, maturity_value, row, binary_value, edge_value in zip(
                training.index, maturity, feature_values, binary, edge
            )
        ],
    }
    return _sha256_json(payload)


def _model_hash(model: DirectEdgeGAM) -> str:
    value = getattr(model, "model_sha256", None)
    if not _is_sha256(value):
        raise ValueError("Direct-edge model returned an invalid SHA-256")
    return str(value)


def _fit_policy(
    frame: pd.DataFrame,
    *,
    model_family: str,
    candidate: DirectEdgeCandidate,
    boundary: pd.Timestamp,
) -> FrozenSectorBreadthPolicy:
    names = _feature_names(model_family)
    training = _training_rows(
        frame, model_family=model_family, boundary=boundary
    )
    training_sha256 = _training_hash(
        training, model_family=model_family, boundary=boundary
    )
    model = DirectEdgeGAM().fit(
        training.loc[:, list(names)],
        training["cash_beats_long_10bps"].to_numpy(dtype=float),
        training["cash_active_log_edge_10bps"].to_numpy(dtype=float),
        feature_names=names,
        fit_metadata=_fit_metadata(training),
    )
    binary = training["cash_beats_long_10bps"].to_numpy(dtype=float)
    edge = training["cash_active_log_edge_10bps"].to_numpy(dtype=float)
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    return FrozenSectorBreadthPolicy(
        model_family=model_family,
        candidate=candidate,
        training_boundary=boundary,
        feature_names=names,
        model=model,
        model_sha256=_model_hash(model),
        training_sha256=training_sha256,
        training_count=int(len(training)),
        training_positive_count=int(binary.sum()),
        training_max_decision_date=training.index.max(),
        training_max_label_maturity_date=maturity.max(),
        baseline_cash_win_probability=float(binary.mean()),
        baseline_mean_net_edge=float(edge.mean()),
    )


def _predict_components(
    model: DirectEdgeGAM,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    raw = model.predict_components(features)
    if not isinstance(raw, Mapping):
        raise ValueError("Direct-edge prediction components must be a mapping")
    try:
        probability = np.asarray(raw["cash_beats_long_probability"], dtype=float)
        edge = np.asarray(raw["expected_edge_10bps"], dtype=float)
    except KeyError as exc:
        raise ValueError("Direct-edge prediction component is missing") from exc
    if probability.shape != (len(features),) or edge.shape != (len(features),):
        raise ValueError("Direct-edge prediction components have invalid shapes")
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(edge).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise ValueError("Direct-edge prediction components are invalid")
    return probability, edge


def _prediction_vectors(
    rows: pd.DataFrame,
    policy: FrozenSectorBreadthPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    ready = _ready_rows(rows, policy.model_family)
    probability = np.full(len(rows), np.nan, dtype=float)
    edge = np.full(len(rows), np.nan, dtype=float)
    if bool(ready.any()):
        predicted_probability, predicted_edge = _predict_components(
            policy.model,
            rows.loc[ready, list(policy.feature_names)],
        )
        probability[ready] = predicted_probability
        edge[ready] = predicted_edge
    return probability, edge


def _require_complete_years(
    index: pd.DatetimeIndex,
    *,
    first_year: int,
    last_year: int,
    period_name: str,
) -> None:
    missing = [year for year in range(first_year, last_year + 1) if year not in index.year]
    if missing:
        raise ValueError(
            f"Incomplete {period_name}; missing decision years: {missing}"
        )


def _write_policy_metadata(
    output: pd.DataFrame,
    *,
    prefix: str,
    policy: FrozenSectorBreadthPolicy,
) -> None:
    output[f"{prefix}_model_sha256"] = policy.model_sha256
    output[f"{prefix}_training_sha256"] = policy.training_sha256
    output[f"{prefix}_training_count"] = policy.training_count
    output[f"{prefix}_training_positive_count"] = policy.training_positive_count
    output[f"{prefix}_training_max_decision_date"] = (
        policy.training_max_decision_date
    )
    output[f"{prefix}_training_max_label_maturity_date"] = (
        policy.training_max_label_maturity_date
    )
    output[f"{prefix}_baseline_cash_win_probability"] = (
        policy.baseline_cash_win_probability
    )
    output[f"{prefix}_baseline_mean_net_edge"] = policy.baseline_mean_net_edge


def _predict_development_fold(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
) -> pd.DataFrame:
    fold_rows = frame.loc[fold.contains(frame.index)].copy()
    _require_complete_years(
        fold_rows.index,
        first_year=fold.first_year,
        last_year=fold.last_year,
        period_name=f"fixed fold {fold.fold_id}",
    )
    boundary = pd.Timestamp(f"{fold.first_year}-01-01")
    neutral_candidate = DIRECT_EDGE_CANDIDATES[0]
    policies = {
        family: _fit_policy(
            frame,
            model_family=family,
            candidate=neutral_candidate,
            boundary=boundary,
        )
        for family in SECTOR_BREADTH_MODEL_FAMILIES
    }
    output = pd.DataFrame(index=fold_rows.index)
    output.index.name = "decision_date"
    output["fold_id"] = fold.fold_id
    output["fold_training_boundary"] = boundary
    output["fold_first_decision_date"] = fold_rows.index.min()
    output["fold_last_decision_date"] = fold_rows.index.max()
    for family, policy in policies.items():
        probability, edge = _prediction_vectors(fold_rows, policy)
        output[f"{family}_cash_win_probability"] = probability
        output[f"{family}_expected_net_edge"] = edge
        _write_policy_metadata(output, prefix=family, policy=policy)
    for name in (*_READINESS_COLUMNS, "sector_breadth_status"):
        output[name] = fold_rows[name]
    for name in DIRECT_EDGE_LABEL_COLUMNS:
        output[name] = fold_rows[name]
    return output


def validate_sector_breadth_binary_exposure(
    predictions: pd.DataFrame,
    *,
    target_columns: Sequence[str] | None = None,
) -> None:
    """Refuse fractional, leveraged, short, or internally inconsistent CASH state."""

    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions must be a nonempty DataFrame")
    if target_columns is None:
        target_columns = tuple(
            name
            for name in predictions.columns
            if name == "cash_target" or name.startswith("cash_target_")
        )
    if not target_columns:
        raise ValueError("No CASH exposure columns are available to validate")
    for target_name in target_columns:
        if target_name not in predictions:
            raise ValueError(f"Missing CASH exposure column: {target_name}")
        if target_name == "cash_target":
            start_name = "cash_block_start"
        else:
            start_name = target_name.replace("cash_target_", "cash_block_start_", 1)
        if start_name not in predictions:
            raise ValueError(f"Missing CASH block-start column: {start_name}")
        target = pd.to_numeric(predictions[target_name], errors="coerce").to_numpy(
            dtype=float
        )
        starts = pd.to_numeric(predictions[start_name], errors="coerce").to_numpy(
            dtype=float
        )
        if (
            not np.isfinite(target).all()
            or not np.isfinite(starts).all()
            or not np.isin(target, (0.0, 1.0)).all()
            or not np.isin(starts, (0.0, 1.0)).all()
        ):
            raise ValueError("CASH exposure and block starts must be exactly binary")
        expected_target, expected_starts = five_session_cash_policy(
            starts.astype(bool)
        )
        if not np.array_equal(starts.astype(np.int8), expected_starts) or not np.array_equal(
            target.astype(np.int8), expected_target
        ):
            raise ValueError("CASH exposure violates the five-session non-overlap policy")


def build_sector_breadth_walkforward_predictions(
    feature_label_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build exact 2005--2018 OOF predictions for all eight frozen candidates."""

    frame = _canonical_frame(feature_label_frame)
    output = pd.concat(
        [
            _predict_development_fold(frame, fold)
            for fold in SECTOR_BREADTH_WALK_FORWARD_FOLDS
        ],
        axis=0,
    ).sort_index(kind="mergesort")
    if output.index.has_duplicates:
        raise RuntimeError("Fixed sector-breadth development folds overlap")
    _require_complete_years(
        output.index,
        first_year=2005,
        last_year=2018,
        period_name="sector-breadth development",
    )
    for family in SECTOR_BREADTH_MODEL_FAMILIES:
        probability = output[f"{family}_cash_win_probability"].to_numpy(dtype=float)
        edge = output[f"{family}_expected_net_edge"].to_numpy(dtype=float)
        finite = np.isfinite(probability) & np.isfinite(edge)
        for candidate in DIRECT_EDGE_CANDIDATES:
            triggers = (
                finite
                & (probability >= candidate.probability_gate)
                & (edge >= candidate.expected_edge_gate)
            )
            target, starts = five_session_cash_policy(triggers)
            output[
                sector_breadth_candidate_cash_target_column(
                    family, candidate.candidate_id
                )
            ] = target
            output[
                sector_breadth_candidate_cash_block_start_column(
                    family, candidate.candidate_id
                )
            ] = starts
    validate_sector_breadth_binary_exposure(output)
    return output


def fit_intermediate_sector_breadth_policy(
    feature_label_frame: pd.DataFrame,
    *,
    selected_family: str,
    selected_candidate: DirectEdgeCandidate | str,
) -> FrozenSectorBreadthPolicy:
    """Fit once using only labels matured by the end of 2018."""

    frame = _canonical_frame(feature_label_frame)
    return _fit_policy(
        frame,
        model_family=selected_family,
        candidate=_candidate_from_value(selected_candidate),
        boundary=INTERMEDIATE_TRAINING_BOUNDARY,
    )


def predict_intermediate_sector_breadth_validation(
    feature_label_frame: pd.DataFrame,
    policy: FrozenSectorBreadthPolicy,
) -> pd.DataFrame:
    """Use one unchanged 2018-boundary policy over all of 2019--2023."""

    if not isinstance(policy, FrozenSectorBreadthPolicy):
        raise TypeError("policy must be a FrozenSectorBreadthPolicy")
    if policy.training_boundary != INTERMEDIATE_TRAINING_BOUNDARY:
        raise ValueError("Intermediate validation requires the 2018-boundary policy")
    frame = _canonical_frame(feature_label_frame)
    period = frame.loc[
        (frame.index >= INTERMEDIATE_FIRST_DECISION)
        & (frame.index <= INTERMEDIATE_LAST_DECISION)
    ].copy()
    _require_complete_years(
        period.index,
        first_year=2019,
        last_year=2023,
        period_name="frozen intermediate validation",
    )
    probability, edge = _prediction_vectors(period, policy)
    finite = np.isfinite(probability) & np.isfinite(edge)
    triggers = (
        finite
        & (probability >= policy.candidate.probability_gate)
        & (edge >= policy.candidate.expected_edge_gate)
    )
    target, starts = five_session_cash_policy(triggers)
    output = pd.DataFrame(index=period.index)
    output.index.name = "decision_date"
    output["model_family"] = policy.model_family
    output["candidate_id"] = policy.candidate_id
    output["gate_id"] = policy.candidate.candidate_id
    output["probability_gate"] = float(policy.candidate.probability_gate)
    output["expected_edge_gate"] = float(policy.candidate.expected_edge_gate)
    output["training_boundary"] = policy.training_boundary
    output["model_sha256"] = policy.model_sha256
    output["training_sha256"] = policy.training_sha256
    output["training_count"] = policy.training_count
    output["training_positive_count"] = policy.training_positive_count
    output["training_max_decision_date"] = policy.training_max_decision_date
    output["training_max_label_maturity_date"] = (
        policy.training_max_label_maturity_date
    )
    output["baseline_cash_win_probability"] = (
        policy.baseline_cash_win_probability
    )
    output["baseline_mean_net_edge"] = policy.baseline_mean_net_edge
    output["cash_win_probability"] = probability
    output["expected_net_edge"] = edge
    output["cash_target"] = target
    output["cash_block_start"] = starts
    for name in (*_READINESS_COLUMNS, "sector_breadth_status"):
        output[name] = period[name]
    for name in DIRECT_EDGE_LABEL_COLUMNS:
        output[name] = period[name]
    validate_sector_breadth_binary_exposure(
        output, target_columns=("cash_target",)
    )
    return output


def fit_final_sector_breadth_policy(
    feature_label_frame: pd.DataFrame,
    *,
    selected_policy: FrozenSectorBreadthPolicy,
) -> FrozenSectorBreadthPolicy:
    """Refit the unchanged selected specification through the 2023 boundary.

    No post-2023 row is accepted and no prediction is made here.  Requiring
    the frozen intermediate object prevents callers from silently substituting
    a different family or gate after observing 2019--2023.
    """

    if not isinstance(selected_policy, FrozenSectorBreadthPolicy):
        raise TypeError("selected_policy must be a FrozenSectorBreadthPolicy")
    if selected_policy.training_boundary != INTERMEDIATE_TRAINING_BOUNDARY:
        raise ValueError("Final refit requires the frozen intermediate policy")
    frame = _canonical_frame(feature_label_frame)
    return _fit_policy(
        frame,
        model_family=selected_policy.model_family,
        candidate=selected_policy.candidate,
        boundary=FINAL_TRAINING_BOUNDARY,
    )


__all__ = [
    "DEVELOPMENT_FIRST_DECISION",
    "DEVELOPMENT_LAST_DECISION",
    "DIRECT_EDGE_CANDIDATES",
    "FINAL_TRAINING_BOUNDARY",
    "FrozenSectorBreadthPolicy",
    "INTERMEDIATE_FIRST_DECISION",
    "INTERMEDIATE_LAST_DECISION",
    "INTERMEDIATE_TRAINING_BOUNDARY",
    "MODEL_FAMILIES",
    "SECTOR_BREADTH_MODEL_FAMILIES",
    "SECTOR_BREADTH_WALK_FORWARD_FOLDS",
    "build_sector_breadth_walkforward_predictions",
    "fit_final_sector_breadth_policy",
    "fit_intermediate_sector_breadth_policy",
    "predict_intermediate_sector_breadth_validation",
    "sector_breadth_candidate_cash_block_start_column",
    "sector_breadth_candidate_cash_target_column",
    "validate_sector_breadth_binary_exposure",
]
