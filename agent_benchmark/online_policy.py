"""Deterministic, point-in-time online policy support for AAPL.

This module deliberately has no benchmark-engine or model-provider dependency.  It
turns matured price paths into structured LONG/CASH/SHORT counterfactual lessons
and uses those lessons in a lightweight local empirical-Bayes long-vs-cash model.

The estimator is designed for test-then-train operation:

1. make a decision from a :class:`PointInTimeSnapshot`;
2. wait until the configured price horizon has elapsed;
3. create and append a :class:`MaturedLesson`;
4. allow later snapshots (and only later snapshots) to use that lesson.

No LLM, network call, paid API, NumPy, pandas, or scikit-learn is required.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


Action = Literal["LONG", "CASH", "SHORT"]
Recommendation = Literal["BUY_ALL", "CASH_ALL", "HOLD"]
TimestampLike = str | date | datetime

SERIALIZATION_VERSION = 1
MODEL_METHOD = "recency_weighted_local_empirical_bayes_v1"
_BLOCKED_FEATURE_PREFIXES = (
    "future_",
    "forward_",
    "lead_",
    "outcome_",
    "target_",
    "label_",
)


def _as_utc(value: TimestampLike, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} cannot be empty")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    else:
        raise TypeError(f"{field_name} must be a string, date, or datetime")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp(value: TimestampLike, *, field_name: str) -> str:
    parsed = _as_utc(value, field_name=field_name)
    return parsed.isoformat().replace("+00:00", "Z")


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _bounded_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _days_between(start: str, end: str) -> float:
    seconds = (_as_utc(end, field_name="end") - _as_utc(start, field_name="start")).total_seconds()
    return seconds / 86_400.0


@dataclass(frozen=True)
class PointInTimeSnapshot:
    """Numerical features that were available before a decision.

    ``as_of_timestamp`` is the common knowledge cutoff.  Optional per-feature
    timestamps make provenance auditable and are rejected if any feature became
    available after that cutoff.  Unspecified feature timestamps default to the
    common cutoff.
    """

    symbol: str
    decision_timestamp: TimestampLike
    as_of_timestamp: TimestampLike
    features: Mapping[str, float]
    feature_timestamps: Mapping[str, TimestampLike] = field(default_factory=dict)

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol cannot be empty")
        decision_timestamp = _timestamp(self.decision_timestamp, field_name="decision_timestamp")
        as_of_timestamp = _timestamp(self.as_of_timestamp, field_name="as_of_timestamp")
        if _as_utc(as_of_timestamp, field_name="as_of_timestamp") > _as_utc(
            decision_timestamp, field_name="decision_timestamp"
        ):
            raise ValueError("as_of_timestamp cannot be after decision_timestamp")

        raw_features = dict(self.features)
        if not raw_features:
            raise ValueError("features cannot be empty")
        raw_timestamps = dict(self.feature_timestamps)
        if any(not isinstance(name, str) for name in raw_features):
            raise ValueError("feature names must be strings")
        if any(not isinstance(name, str) for name in raw_timestamps):
            raise ValueError("feature timestamp names must be strings")
        extras = set(raw_timestamps) - set(raw_features)
        if extras:
            raise ValueError(f"feature_timestamps contains unknown features: {sorted(extras)}")

        features: dict[str, float] = {}
        feature_timestamps: dict[str, str] = {}
        for raw_name in sorted(raw_features):
            name = str(raw_name).strip()
            if not name:
                raise ValueError("feature names cannot be empty")
            lowered = name.casefold()
            if lowered.startswith(_BLOCKED_FEATURE_PREFIXES):
                raise ValueError(
                    f"feature {name!r} looks outcome-derived; point-in-time features cannot "
                    "use future/forward/lead/outcome/target/label prefixes"
                )
            features[name] = _finite_float(raw_features[raw_name], field_name=f"features[{name!r}]")
            observed_at = _timestamp(
                raw_timestamps.get(raw_name, as_of_timestamp),
                field_name=f"feature_timestamps[{name!r}]",
            )
            if _as_utc(observed_at, field_name="feature timestamp") > _as_utc(
                as_of_timestamp, field_name="as_of_timestamp"
            ):
                raise ValueError(f"feature {name!r} was not available at as_of_timestamp")
            feature_timestamps[name] = observed_at

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "decision_timestamp", decision_timestamp)
        object.__setattr__(self, "as_of_timestamp", as_of_timestamp)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_timestamps", feature_timestamps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision_timestamp": self.decision_timestamp,
            "as_of_timestamp": self.as_of_timestamp,
            "features": dict(self.features),
            "feature_timestamps": dict(self.feature_timestamps),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PointInTimeSnapshot":
        return cls(
            symbol=payload["symbol"],
            decision_timestamp=payload["decision_timestamp"],
            as_of_timestamp=payload["as_of_timestamp"],
            features=payload["features"],
            feature_timestamps=payload.get("feature_timestamps", {}),
        )


@dataclass(frozen=True)
class PricePoint:
    timestamp: TimestampLike
    price: float

    def __post_init__(self) -> None:
        timestamp = _timestamp(self.timestamp, field_name="price timestamp")
        price = _finite_float(self.price, field_name="price")
        if price <= 0:
            raise ValueError("price must be greater than zero")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "price", price)

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self.timestamp, "price": self.price}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PricePoint":
        return cls(timestamp=payload["timestamp"], price=payload["price"])


@dataclass(frozen=True)
class CounterfactualCostModel:
    """Return-level cost assumptions for a long-default policy.

    Turnover is measured in one-way notional units.  The defaults therefore
    charge no trade for continuing to hold LONG, two units for temporarily
    moving LONG -> CASH -> LONG, and four for LONG -> SHORT -> LONG.  The
    transaction rate is applied once per turnover unit.
    """

    transaction_cost_bps: float = 5.0
    long_turnover: float = 0.0
    cash_turnover: float = 2.0
    short_turnover: float = 4.0
    annual_short_borrow_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "transaction_cost_bps",
            "long_turnover",
            "cash_turnover",
            "short_turnover",
            "annual_short_borrow_bps",
        ):
            value = _finite_float(getattr(self, name), field_name=name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float]:
        return {
            "transaction_cost_bps": self.transaction_cost_bps,
            "long_turnover": self.long_turnover,
            "cash_turnover": self.cash_turnover,
            "short_turnover": self.short_turnover,
            "annual_short_borrow_bps": self.annual_short_borrow_bps,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualCostModel":
        return cls(**dict(payload))


@dataclass(frozen=True)
class ActionOutcome:
    """Gross and after-cost return for one counterfactual action."""

    action: Action
    gross_return: float
    transaction_cost: float
    financing_cost: float
    net_return: float

    def __post_init__(self) -> None:
        if self.action not in ("LONG", "CASH", "SHORT"):
            raise ValueError(f"unsupported action: {self.action!r}")
        gross_return = _finite_float(self.gross_return, field_name="gross_return")
        transaction_cost = _finite_float(self.transaction_cost, field_name="transaction_cost")
        financing_cost = _finite_float(self.financing_cost, field_name="financing_cost")
        net_return = _finite_float(self.net_return, field_name="net_return")
        if transaction_cost < 0 or financing_cost < 0:
            raise ValueError("costs cannot be negative")
        expected_net = gross_return - transaction_cost - financing_cost
        if not math.isclose(net_return, expected_net, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("net_return must equal gross_return minus all costs")
        object.__setattr__(self, "gross_return", gross_return)
        object.__setattr__(self, "transaction_cost", transaction_cost)
        object.__setattr__(self, "financing_cost", financing_cost)
        object.__setattr__(self, "net_return", net_return)

    @property
    def total_cost(self) -> float:
        return self.transaction_cost + self.financing_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "gross_return": self.gross_return,
            "transaction_cost": self.transaction_cost,
            "financing_cost": self.financing_cost,
            "total_cost": self.total_cost,
            "net_return": self.net_return,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionOutcome":
        return cls(
            action=payload["action"],
            gross_return=payload["gross_return"],
            transaction_cost=payload["transaction_cost"],
            financing_cost=payload["financing_cost"],
            net_return=payload["net_return"],
        )


@dataclass(frozen=True)
class CounterfactualOutcomes:
    """Matured LONG/CASH/SHORT returns, all expressed after costs."""

    long: ActionOutcome
    cash: ActionOutcome
    short: ActionOutcome

    def __post_init__(self) -> None:
        if (self.long.action, self.cash.action, self.short.action) != ("LONG", "CASH", "SHORT"):
            raise ValueError("counterfactual outcomes must be ordered as LONG, CASH, SHORT")

    @property
    def cash_active_return(self) -> float:
        """After-cost CASH return minus after-cost buy-and-hold return."""

        return self.cash.net_return - self.long.net_return

    @property
    def short_active_return(self) -> float:
        return self.short.net_return - self.long.net_return

    @property
    def cash_beats_long(self) -> bool:
        return self.cash_active_return > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "LONG": self.long.to_dict(),
            "CASH": self.cash.to_dict(),
            "SHORT": self.short.to_dict(),
            "cash_active_return": self.cash_active_return,
            "short_active_return": self.short_active_return,
            "cash_beats_long": self.cash_beats_long,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualOutcomes":
        return cls(
            long=ActionOutcome.from_dict(payload["LONG"]),
            cash=ActionOutcome.from_dict(payload["CASH"]),
            short=ActionOutcome.from_dict(payload["SHORT"]),
        )


@dataclass(frozen=True)
class MaturedLesson:
    """A point-in-time feature state whose future outcome is now knowable."""

    lesson_id: str
    snapshot: PointInTimeSnapshot
    entry_timestamp: TimestampLike
    outcome_timestamp: TimestampLike
    entry_price: float
    exit_price: float
    price_observations: int
    horizon_days: float
    outcomes: CounterfactualOutcomes
    cost_model: CounterfactualCostModel
    available_timestamp: TimestampLike | None = None

    def __post_init__(self) -> None:
        lesson_id = str(self.lesson_id).strip()
        if not lesson_id:
            raise ValueError("lesson_id cannot be empty")
        entry_timestamp = _timestamp(self.entry_timestamp, field_name="entry_timestamp")
        outcome_timestamp = _timestamp(self.outcome_timestamp, field_name="outcome_timestamp")
        if _as_utc(entry_timestamp, field_name="entry_timestamp") < _as_utc(
            self.snapshot.decision_timestamp, field_name="decision_timestamp"
        ):
            raise ValueError("entry_timestamp cannot precede the decision")
        if _as_utc(outcome_timestamp, field_name="outcome_timestamp") <= _as_utc(
            entry_timestamp, field_name="entry_timestamp"
        ):
            raise ValueError("outcome_timestamp must be after entry_timestamp")
        entry_price = _finite_float(self.entry_price, field_name="entry_price")
        exit_price = _finite_float(self.exit_price, field_name="exit_price")
        if entry_price <= 0 or exit_price <= 0:
            raise ValueError("entry_price and exit_price must be greater than zero")
        observations = int(self.price_observations)
        if observations < 2:
            raise ValueError("price_observations must be at least two")
        horizon_days = _finite_float(self.horizon_days, field_name="horizon_days")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        actual_days = _days_between(entry_timestamp, outcome_timestamp)
        if not math.isclose(horizon_days, actual_days, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("horizon_days must match entry and outcome timestamps")
        available_timestamp = _timestamp(
            self.available_timestamp or outcome_timestamp,
            field_name="available_timestamp",
        )
        if _as_utc(available_timestamp, field_name="available_timestamp") < _as_utc(
            outcome_timestamp, field_name="outcome_timestamp"
        ):
            raise ValueError("available_timestamp cannot precede the outcome")
        object.__setattr__(self, "lesson_id", lesson_id)
        object.__setattr__(self, "entry_timestamp", entry_timestamp)
        object.__setattr__(self, "outcome_timestamp", outcome_timestamp)
        object.__setattr__(self, "entry_price", entry_price)
        object.__setattr__(self, "exit_price", exit_price)
        object.__setattr__(self, "price_observations", observations)
        object.__setattr__(self, "horizon_days", horizon_days)
        object.__setattr__(self, "available_timestamp", available_timestamp)

    @property
    def knowledge_timestamp(self) -> str:
        """The first time this outcome may be used by a later decision."""

        return str(self.available_timestamp)

    @property
    def features(self) -> Mapping[str, float]:
        return self.snapshot.features

    @property
    def risk_off_label(self) -> int:
        return int(self.outcomes.cash_beats_long)

    @property
    def cash_active_return(self) -> float:
        return self.outcomes.cash_active_return

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "snapshot": self.snapshot.to_dict(),
            "entry_timestamp": self.entry_timestamp,
            "outcome_timestamp": self.outcome_timestamp,
            "knowledge_timestamp": self.knowledge_timestamp,
            "available_timestamp": self.available_timestamp,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "price_observations": self.price_observations,
            "horizon_days": self.horizon_days,
            "risk_off_label": self.risk_off_label,
            "cash_active_return": self.cash_active_return,
            "outcomes": self.outcomes.to_dict(),
            "cost_model": self.cost_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MaturedLesson":
        return cls(
            lesson_id=payload["lesson_id"],
            snapshot=PointInTimeSnapshot.from_dict(payload["snapshot"]),
            entry_timestamp=payload["entry_timestamp"],
            outcome_timestamp=payload["outcome_timestamp"],
            entry_price=payload["entry_price"],
            exit_price=payload["exit_price"],
            price_observations=payload["price_observations"],
            horizon_days=payload["horizon_days"],
            outcomes=CounterfactualOutcomes.from_dict(payload["outcomes"]),
            cost_model=CounterfactualCostModel.from_dict(payload["cost_model"]),
            available_timestamp=payload.get("available_timestamp") or payload.get("knowledge_timestamp"),
        )


def create_matured_lesson(
    snapshot: PointInTimeSnapshot,
    price_path: Sequence[PricePoint],
    *,
    cost_model: CounterfactualCostModel | None = None,
    knowledge_timestamp: TimestampLike | None = None,
) -> MaturedLesson:
    """Create an after-cost counterfactual lesson from a completed price path.

    The first point is the executable entry price and the last point is the
    matured outcome price.  Intermediate points are accepted for auditability
    and future risk statistics, but are never copied into model features.  The
    caller should supply dividend-adjusted or total-return prices when the
    benchmark being compared also includes distributions.
    """

    points = [point if isinstance(point, PricePoint) else PricePoint.from_dict(point) for point in price_path]
    if len(points) < 2:
        raise ValueError("price_path must contain an entry and at least one later price")
    for previous, current in zip(points, points[1:]):
        if _as_utc(current.timestamp, field_name="price timestamp") <= _as_utc(
            previous.timestamp, field_name="price timestamp"
        ):
            raise ValueError("price_path timestamps must be strictly increasing")
    entry = points[0]
    outcome = points[-1]
    if _as_utc(entry.timestamp, field_name="entry timestamp") < _as_utc(
        snapshot.decision_timestamp, field_name="decision_timestamp"
    ):
        raise ValueError("the entry price cannot be known before the decision is made")

    costs = cost_model or CounterfactualCostModel()
    gross_long = outcome.price / entry.price - 1.0
    transaction_rate = costs.transaction_cost_bps / 10_000.0
    horizon_days = _days_between(entry.timestamp, outcome.timestamp)
    short_borrow = costs.annual_short_borrow_bps / 10_000.0 * horizon_days / 365.25

    long_transaction = transaction_rate * costs.long_turnover
    cash_transaction = transaction_rate * costs.cash_turnover
    short_transaction = transaction_rate * costs.short_turnover
    long = ActionOutcome(
        action="LONG",
        gross_return=gross_long,
        transaction_cost=long_transaction,
        financing_cost=0.0,
        net_return=gross_long - long_transaction,
    )
    cash = ActionOutcome(
        action="CASH",
        gross_return=0.0,
        transaction_cost=cash_transaction,
        financing_cost=0.0,
        net_return=-cash_transaction,
    )
    short = ActionOutcome(
        action="SHORT",
        gross_return=-gross_long,
        transaction_cost=short_transaction,
        financing_cost=short_borrow,
        net_return=-gross_long - short_transaction - short_borrow,
    )
    outcomes = CounterfactualOutcomes(long=long, cash=cash, short=short)
    identity = {
        "snapshot": snapshot.to_dict(),
        "entry": entry.to_dict(),
        "outcome": outcome.to_dict(),
        "cost_model": costs.to_dict(),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return MaturedLesson(
        lesson_id=f"online:{snapshot.symbol}:{digest}",
        snapshot=snapshot,
        entry_timestamp=entry.timestamp,
        outcome_timestamp=outcome.timestamp,
        entry_price=entry.price,
        exit_price=outcome.price,
        price_observations=len(points),
        horizon_days=horizon_days,
        outcomes=outcomes,
        cost_model=costs,
        available_timestamp=knowledge_timestamp or outcome.timestamp,
    )


@dataclass(frozen=True)
class RiskOffEstimatorConfig:
    """Generic statistical controls; none are tied to a calendar year."""

    symbol: str = "AAPL"
    max_neighbors: int = 64
    min_samples: int = 20
    min_neighbor_separation_days: int = 0
    prior_strength: float = 4.0
    risk_off_probability: float = 0.60
    min_confidence: float = 0.30
    min_active_return: float = 0.0
    confidence_z: float = 1.645
    recency_half_life_days: float = 730.5
    min_feature_overlap: float = 0.50

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol cannot be empty")
        max_neighbors = int(self.max_neighbors)
        min_samples = int(self.min_samples)
        min_neighbor_separation_days = int(self.min_neighbor_separation_days)
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be at least one")
        if min_samples < 1:
            raise ValueError("min_samples must be at least one")
        if min_samples > max_neighbors:
            raise ValueError("min_samples cannot exceed max_neighbors")
        if min_neighbor_separation_days < 0:
            raise ValueError("min_neighbor_separation_days cannot be negative")
        prior_strength = _finite_float(self.prior_strength, field_name="prior_strength")
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        risk_off_probability = _finite_float(
            self.risk_off_probability, field_name="risk_off_probability"
        )
        if not 0.5 < risk_off_probability < 1.0:
            raise ValueError("risk_off_probability must be between 0.5 and 1")
        min_confidence = _finite_float(self.min_confidence, field_name="min_confidence")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        min_active_return = _finite_float(self.min_active_return, field_name="min_active_return")
        if min_active_return < 0:
            raise ValueError("min_active_return cannot be negative")
        confidence_z = _finite_float(self.confidence_z, field_name="confidence_z")
        if confidence_z <= 0:
            raise ValueError("confidence_z must be positive")
        recency_half_life_days = _finite_float(
            self.recency_half_life_days, field_name="recency_half_life_days"
        )
        if recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive")
        min_feature_overlap = _finite_float(
            self.min_feature_overlap, field_name="min_feature_overlap"
        )
        if not 0.0 < min_feature_overlap <= 1.0:
            raise ValueError("min_feature_overlap must be in (0, 1]")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "max_neighbors", max_neighbors)
        object.__setattr__(self, "min_samples", min_samples)
        object.__setattr__(self, "min_neighbor_separation_days", min_neighbor_separation_days)
        object.__setattr__(self, "prior_strength", prior_strength)
        object.__setattr__(self, "risk_off_probability", risk_off_probability)
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "min_active_return", min_active_return)
        object.__setattr__(self, "confidence_z", confidence_z)
        object.__setattr__(self, "recency_half_life_days", recency_half_life_days)
        object.__setattr__(self, "min_feature_overlap", min_feature_overlap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "max_neighbors": self.max_neighbors,
            "min_samples": self.min_samples,
            "min_neighbor_separation_days": self.min_neighbor_separation_days,
            "prior_strength": self.prior_strength,
            "risk_off_probability": self.risk_off_probability,
            "min_confidence": self.min_confidence,
            "min_active_return": self.min_active_return,
            "confidence_z": self.confidence_z,
            "recency_half_life_days": self.recency_half_life_days,
            "min_feature_overlap": self.min_feature_overlap,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RiskOffEstimatorConfig":
        return cls(**dict(payload))


@dataclass(frozen=True)
class DecisionSupport:
    """Auditable output for a long-default BUY/CASH/HOLD decision.

    ``expected_active_return`` and its bounds are CASH minus LONG after costs.
    Consequently, ``loss_probability`` is the local empirical estimate that a
    CASH decision would fail to outperform remaining LONG; it is not a forecast
    of AAPL's absolute probability of a negative return.
    """

    symbol: str
    as_of_timestamp: str
    expected_active_return: float
    loss_probability: float
    confidence: float
    lower_bound: float
    upper_bound: float
    cash_outperformance_probability: float
    probability_lower_bound: float
    probability_upper_bound: float
    sample_size: int
    effective_sample_size: float
    eligible_lesson_count: int
    recommended_action: Recommendation
    method: str = MODEL_METHOD

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of_timestamp": self.as_of_timestamp,
            "expected_active_return": self.expected_active_return,
            "loss_probability": self.loss_probability,
            "confidence": self.confidence,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "cash_outperformance_probability": self.cash_outperformance_probability,
            "probability_lower_bound": self.probability_lower_bound,
            "probability_upper_bound": self.probability_upper_bound,
            "sample_size": self.sample_size,
            "effective_sample_size": self.effective_sample_size,
            "eligible_lesson_count": self.eligible_lesson_count,
            "recommended_action": self.recommended_action,
            "method": self.method,
        }


@dataclass(frozen=True)
class _Neighbor:
    lesson: MaturedLesson
    distance: float
    similarity: float
    overlap: float


class CalibratedOnlineRiskOffEstimator:
    """Deterministic local empirical-Bayes estimator for CASH vs LONG.

    Similar historical states are selected after feature standardization.  Their
    binary CASH-beats-LONG labels update a beta prior, while their continuous
    after-cost active returns estimate edge and uncertainty.  The probability is
    a smoothed local observed frequency rather than a held-out calibrated claim
    or an unbounded model score.  Live promotion still requires separate
    walk-forward reliability and coverage checks.

    Lessons are appended incrementally in maturation order.  At prediction time
    they are still filtered by the snapshot's knowledge cutoff, so re-running an
    older point in time cannot accidentally consume a later outcome.
    """

    def __init__(self, config: RiskOffEstimatorConfig | None = None):
        self.config = config or RiskOffEstimatorConfig()
        self._lessons: list[MaturedLesson] = []
        self._lesson_ids: set[str] = set()

    @property
    def lessons(self) -> tuple[MaturedLesson, ...]:
        return tuple(self._lessons)

    @property
    def last_knowledge_timestamp(self) -> str | None:
        if not self._lessons:
            return None
        return self._lessons[-1].knowledge_timestamp

    def update(self, lesson: MaturedLesson) -> None:
        """Append one newly matured lesson without replaying older outcomes."""

        self.update_many([lesson])

    def update_many(self, lessons: Iterable[MaturedLesson]) -> None:
        """Atomically append lessons, requiring online chronological order."""

        pending = list(lessons)
        known_ids = set(self._lesson_ids)
        last_timestamp = self.last_knowledge_timestamp
        for lesson in pending:
            if not isinstance(lesson, MaturedLesson):
                raise TypeError("all updates must be MaturedLesson instances")
            if lesson.snapshot.symbol != self.config.symbol:
                raise ValueError(
                    f"lesson symbol {lesson.snapshot.symbol!r} does not match estimator "
                    f"symbol {self.config.symbol!r}"
                )
            if lesson.lesson_id in known_ids:
                raise ValueError(f"duplicate lesson_id: {lesson.lesson_id}")
            if last_timestamp is not None and _as_utc(
                lesson.knowledge_timestamp, field_name="knowledge_timestamp"
            ) < _as_utc(last_timestamp, field_name="last knowledge timestamp"):
                raise ValueError("online lessons must be appended in knowledge-timestamp order")
            known_ids.add(lesson.lesson_id)
            last_timestamp = lesson.knowledge_timestamp
        self._lessons.extend(pending)
        self._lesson_ids = known_ids

    def decision_support(self, snapshot: PointInTimeSnapshot) -> DecisionSupport:
        if snapshot.symbol != self.config.symbol:
            raise ValueError(
                f"snapshot symbol {snapshot.symbol!r} does not match estimator "
                f"symbol {self.config.symbol!r}"
            )
        cutoff = _as_utc(snapshot.as_of_timestamp, field_name="as_of_timestamp")
        visible = [
            lesson
            for lesson in self._lessons
            if _as_utc(lesson.knowledge_timestamp, field_name="knowledge_timestamp") <= cutoff
        ]
        neighbors = self._neighbors(snapshot, visible)
        if not neighbors:
            return DecisionSupport(
                symbol=snapshot.symbol,
                as_of_timestamp=str(snapshot.as_of_timestamp),
                expected_active_return=0.0,
                loss_probability=0.5,
                confidence=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                cash_outperformance_probability=0.5,
                probability_lower_bound=0.0,
                probability_upper_bound=1.0,
                sample_size=0,
                effective_sample_size=0.0,
                eligible_lesson_count=len(visible),
                recommended_action="HOLD",
            )

        all_labels = [lesson.risk_off_label for lesson in visible]
        all_returns = [lesson.cash_active_return for lesson in visible]
        # A uniform beta hyper-prior prevents exact 0/1 base rates.
        global_probability = (sum(all_labels) + 1.0) / (len(all_labels) + 2.0)
        global_mean = sum(all_returns) / len(all_returns)
        global_variance = (
            sum((value - global_mean) ** 2 for value in all_returns) / len(all_returns)
            if len(all_returns) > 1
            else 0.0
        )

        weighted: list[tuple[_Neighbor, float]] = []
        for neighbor in neighbors:
            age_days = max(
                0.0,
                _days_between(neighbor.lesson.knowledge_timestamp, str(snapshot.as_of_timestamp)),
            )
            recency = 0.5 ** (age_days / self.config.recency_half_life_days)
            weighted.append((neighbor, neighbor.similarity * recency))

        weight_sum = sum(weight for _, weight in weighted)
        weight_square_sum = sum(weight * weight for _, weight in weighted)
        effective_n = weight_sum * weight_sum / weight_square_sum if weight_square_sum else 0.0
        successes = sum(weight * neighbor.lesson.risk_off_label for neighbor, weight in weighted)
        alpha = self.config.prior_strength * global_probability + successes
        beta = self.config.prior_strength * (1.0 - global_probability) + weight_sum - successes
        probability = _bounded_probability(alpha / (alpha + beta))
        loss_probability = _bounded_probability(1.0 - probability)
        probability_variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        probability_error = self.config.confidence_z * math.sqrt(max(0.0, probability_variance))
        probability_lower = _bounded_probability(probability - probability_error)
        probability_upper = _bounded_probability(probability + probability_error)

        return_denominator = self.config.prior_strength + weight_sum
        expected_return = (
            self.config.prior_strength * global_mean
            + sum(weight * neighbor.lesson.cash_active_return for neighbor, weight in weighted)
        ) / return_denominator
        variance_numerator = self.config.prior_strength * (
            global_variance + (global_mean - expected_return) ** 2
        ) + sum(
            weight * (neighbor.lesson.cash_active_return - expected_return) ** 2
            for neighbor, weight in weighted
        )
        active_variance = max(0.0, variance_numerator / return_denominator)
        standard_error = math.sqrt(active_variance / max(1.0, effective_n))
        return_error = self.config.confidence_z * standard_error
        lower_bound = expected_return - return_error
        upper_bound = expected_return + return_error

        average_overlap = (
            sum(weight * neighbor.overlap for neighbor, weight in weighted) / weight_sum
            if weight_sum
            else 0.0
        )
        evidence_confidence = weight_sum / (weight_sum + self.config.prior_strength)
        sample_confidence = min(1.0, effective_n / self.config.min_samples)
        confidence = _bounded_probability(evidence_confidence * sample_confidence * average_overlap)

        enough_evidence = (
            len(neighbors) >= self.config.min_samples
            and effective_n >= self.config.min_samples * 0.75
            and confidence >= self.config.min_confidence
        )
        if (
            enough_evidence
            and probability >= self.config.risk_off_probability
            and probability_lower > 0.5
            and lower_bound > self.config.min_active_return
        ):
            recommendation: Recommendation = "CASH_ALL"
        elif (
            enough_evidence
            and probability <= 1.0 - self.config.risk_off_probability
            and probability_upper < 0.5
            and upper_bound < -self.config.min_active_return
        ):
            recommendation = "BUY_ALL"
        else:
            recommendation = "HOLD"

        return DecisionSupport(
            symbol=snapshot.symbol,
            as_of_timestamp=str(snapshot.as_of_timestamp),
            expected_active_return=expected_return,
            loss_probability=loss_probability,
            confidence=confidence,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            cash_outperformance_probability=probability,
            probability_lower_bound=probability_lower,
            probability_upper_bound=probability_upper,
            sample_size=len(neighbors),
            effective_sample_size=effective_n,
            eligible_lesson_count=len(visible),
            recommended_action=recommendation,
        )

    def _neighbors(
        self,
        snapshot: PointInTimeSnapshot,
        visible: Sequence[MaturedLesson],
    ) -> list[_Neighbor]:
        if not visible:
            return []
        scales: dict[str, tuple[float, float]] = {}
        for feature_name in snapshot.features:
            values = [
                lesson.features[feature_name]
                for lesson in visible
                if feature_name in lesson.features
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / len(values)
                if len(values) > 1
                else 0.0
            )
            # A constant historical feature still needs to penalize a different
            # current value.  The scale floor is unit-aware through its mean.
            scale = max(math.sqrt(variance), abs(mean) * 0.05, 1e-9)
            scales[feature_name] = (mean, scale)

        neighbors: list[_Neighbor] = []
        for lesson in visible:
            shared = sorted(set(snapshot.features) & set(lesson.features) & set(scales))
            canonical_feature_count = len(set(snapshot.features) | set(lesson.features))
            overlap = len(shared) / canonical_feature_count if canonical_feature_count else 0.0
            if not shared or overlap < self.config.min_feature_overlap:
                continue
            squared_distance = 0.0
            for name in shared:
                _, scale = scales[name]
                standardized = (snapshot.features[name] - lesson.features[name]) / scale
                squared_distance += min(100.0, standardized * standardized)
            distance = math.sqrt(squared_distance / len(shared))
            similarity = 1.0 / (1.0 + distance * distance)
            neighbors.append(
                _Neighbor(
                    lesson=lesson,
                    distance=distance,
                    similarity=similarity,
                    overlap=overlap,
                )
            )
        def selection_key(item: _Neighbor) -> tuple[float, float, str]:
            age_days = max(
                0.0,
                _days_between(item.lesson.knowledge_timestamp, str(snapshot.as_of_timestamp)),
            )
            recency = 0.5 ** (age_days / self.config.recency_half_life_days)
            combined_weight = item.similarity * recency
            return (-combined_weight, item.distance, item.lesson.lesson_id)

        # Selecting on the same similarity-times-recency weight used by the
        # posterior prevents a large bank of ancient exact matches from
        # crowding all newly matured regimes out of a bounded neighbor set.
        neighbors.sort(key=selection_key)
        separation = int(self.config.min_neighbor_separation_days)
        selected: list[_Neighbor] = []
        for candidate in neighbors:
            candidate_time = candidate.lesson.snapshot.decision_timestamp
            if any(
                (
                    _as_utc(candidate.lesson.entry_timestamp, field_name="entry_timestamp")
                    <= _as_utc(item.lesson.outcome_timestamp, field_name="outcome_timestamp")
                    and _as_utc(item.lesson.entry_timestamp, field_name="entry_timestamp")
                    <= _as_utc(candidate.lesson.outcome_timestamp, field_name="outcome_timestamp")
                )
                or (
                    separation > 0
                    and abs(_days_between(item.lesson.snapshot.decision_timestamp, candidate_time)) < separation
                )
                for item in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= self.config.max_neighbors:
                break
        return selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "serialization_version": SERIALIZATION_VERSION,
            "method": MODEL_METHOD,
            "config": self.config.to_dict(),
            "lessons": [lesson.to_dict() for lesson in self._lessons],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibratedOnlineRiskOffEstimator":
        version = int(payload.get("serialization_version", -1))
        if version != SERIALIZATION_VERSION:
            raise ValueError(
                f"unsupported serialization_version {version}; expected {SERIALIZATION_VERSION}"
            )
        if payload.get("method") != MODEL_METHOD:
            raise ValueError(f"unsupported estimator method: {payload.get('method')!r}")
        estimator = cls(RiskOffEstimatorConfig.from_dict(payload["config"]))
        estimator.update_many(MaturedLesson.from_dict(item) for item in payload.get("lessons", []))
        return estimator

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Atomically persist the complete online state as deterministic JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CalibratedOnlineRiskOffEstimator":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("serialized estimator must be a JSON object")
        return cls.from_dict(payload)


__all__ = [
    "ActionOutcome",
    "CalibratedOnlineRiskOffEstimator",
    "CounterfactualCostModel",
    "CounterfactualOutcomes",
    "DecisionSupport",
    "MaturedLesson",
    "PointInTimeSnapshot",
    "PricePoint",
    "RiskOffEstimatorConfig",
    "create_matured_lesson",
]
