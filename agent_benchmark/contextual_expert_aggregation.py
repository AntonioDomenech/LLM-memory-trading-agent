"""Pure causal contextual expert aggregation for AAPL LONG/CASH advice.

The module owns only deterministic sequential model state.  It performs no
data acquisition, historical-stage scoring, account simulation, LLM call, or
network request.  A caller supplies one already-authorized daily market row
at a time together with the fixed contextual/weak-trend signal membership of
the canonical union opportunity.

Every accepted opportunity creates an action-independent shadow lesson.  Its
entry adjusted open is observed on the next supplied session and its exit
adjusted open on the following session.  Only then may the 10-bps-per-side
CASH edge update later decisions.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence


Action = Literal["LONG", "CASH"]
LearningMode = Literal["causal_online", "frozen_cutoff"]
AblationMode = Literal["full", "global_only", "lifetime_only"]
DateLike = str | date | datetime

CAUSAL_ONLINE_MODE = "causal_online"
FROZEN_CUTOFF_MODE = "frozen_cutoff"
LEARNING_MODES = frozenset({CAUSAL_ONLINE_MODE, FROZEN_CUTOFF_MODE})

FULL_MODE = "full"
GLOBAL_ONLY_MODE = "global_only"
LIFETIME_ONLY_MODE = "lifetime_only"
ABLATION_MODES = frozenset({FULL_MODE, GLOBAL_ONLY_MODE, LIFETIME_ONLY_MODE})

EXPERT_NAMES = (
    "always_long",
    "union_cash",
    "contextual_only",
    "weak_trend_only",
)
FAST_LOOKBACK = 10
SLOW_LOOKBACK = 20
MARKET_STATE_NAMES = (
    "fast_off_slow_off",
    "fast_off_slow_on",
    "fast_on_slow_off",
    "fast_on_slow_on",
)
UNKNOWN_MARKET_STATE = "unknown"

FAST_SCALE = "half_life_8"
MEDIUM_SCALE = "half_life_32"
LIFETIME_SCALE = "lifetime"
SCALE_NAMES = (FAST_SCALE, MEDIUM_SCALE, LIFETIME_SCALE)
SCALE_HALF_LIVES: Mapping[str, float | None] = {
    FAST_SCALE: 8.0,
    MEDIUM_SCALE: 32.0,
    LIFETIME_SCALE: None,
}
SCALE_DISCOUNTS: Mapping[str, float] = {
    name: (1.0 if half_life is None else 2.0 ** (-1.0 / half_life))
    for name, half_life in SCALE_HALF_LIVES.items()
}

PER_SIDE_FRICTION = 0.001
ROUND_TRIP_LOG_FRICTION = math.log(
    (1.0 - PER_SIDE_FRICTION) / (1.0 + PER_SIDE_FRICTION)
)
LESSON_MEMORY_START = date(2000, 1, 1)
LESSON_MEMORY_START_ISO = LESSON_MEMORY_START.isoformat()
STATE_PRIOR_STRENGTH = float(len(EXPERT_NAMES))
LONG_TIE_TOLERANCE = 1e-12
_NORMAL_MIN = sys.float_info.min
_STATE_ULP_TOLERANCE = 32.0

SERIALIZATION_VERSION = 2
MODEL_METHOD = "causal_contextual_expert_aggregation_v1"


def _strict_keys(
    value: Mapping[str, Any], expected: set[str], *, field_name: str
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if set(value) != expected:
        raise ValueError(f"{field_name} has missing or unexpected fields")


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _normal_or_zero(value: float) -> bool:
    return bool(
        math.isfinite(value)
        and (value == 0.0 or abs(value) >= _NORMAL_MIN)
    )


def _checked_normal_or_zero(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not _normal_or_zero(result):
        raise ValueError(
            f"{field_name} must be finite and either zero or normal"
        )
    return result


def _finite_positive(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not _normal_or_zero(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be positive, finite, and normal")
    return result


def _finite_nonnegative(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not _normal_or_zero(result) or result < 0.0:
        raise ValueError(
            f"{field_name} must be finite, non-negative, and normal or zero"
        )
    return result


def _iso_date(value: DateLike, *, field_name: str) -> str:
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} cannot be empty")
        try:
            parsed = date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO calendar date") from exc
    else:
        raise TypeError(f"{field_name} must be a date, datetime, or ISO date")
    return parsed.isoformat()


def _date_value(value: str) -> date:
    return date.fromisoformat(value)


def _rounding_margin(*values: float) -> float:
    """Return a small scale-aware allowance for legitimate float recursion."""

    finite = [abs(float(value)) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    scale = max(finite)
    if scale == 0.0:
        return 0.0
    return _STATE_ULP_TOLERANCE * math.ulp(scale)


def derive_adjusted_open(
    *, aapl_open: float, aapl_close: float, aapl_adj_close: float
) -> float:
    """Derive adjusted open from the exact six-column price contract."""

    raw_open = _finite_positive(aapl_open, field_name="aapl_open")
    raw_close = _finite_positive(aapl_close, field_name="aapl_close")
    adjusted_close = _finite_positive(
        aapl_adj_close, field_name="aapl_adj_close"
    )
    value = raw_open * adjusted_close / raw_close
    return _finite_positive(value, field_name="derived adjusted open")


@dataclass(frozen=True)
class MarketSession:
    """One completed daily row plus canonical fixed-expert membership."""

    session_date: DateLike
    aapl_open: float
    aapl_close: float
    aapl_adj_close: float
    spy_adj_close: float
    qqq_adj_close: float
    contextual_signal: bool = False
    weak_trend_signal: bool = False
    canonical_union_opportunity: bool | None = None
    aapl_adj_open: float | None = None

    def __post_init__(self) -> None:
        session_date = _iso_date(self.session_date, field_name="session_date")
        raw_open = _finite_positive(self.aapl_open, field_name="aapl_open")
        raw_close = _finite_positive(self.aapl_close, field_name="aapl_close")
        adjusted_close = _finite_positive(
            self.aapl_adj_close, field_name="aapl_adj_close"
        )
        spy = _finite_positive(self.spy_adj_close, field_name="spy_adj_close")
        qqq = _finite_positive(self.qqq_adj_close, field_name="qqq_adj_close")
        contextual = _strict_bool(
            self.contextual_signal, field_name="contextual_signal"
        )
        weak = _strict_bool(
            self.weak_trend_signal, field_name="weak_trend_signal"
        )
        opportunity = self.canonical_union_opportunity
        if opportunity is None:
            opportunity = bool(contextual or weak)
        opportunity = _strict_bool(
            opportunity, field_name="canonical_union_opportunity"
        )
        if opportunity and not (contextual or weak):
            raise ValueError(
                "a canonical union opportunity requires a contributing expert"
            )

        derived = derive_adjusted_open(
            aapl_open=raw_open,
            aapl_close=raw_close,
            aapl_adj_close=adjusted_close,
        )
        if self.aapl_adj_open is not None:
            supplied = _finite_positive(
                self.aapl_adj_open, field_name="aapl_adj_open"
            )
            if not math.isclose(
                supplied, derived, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    "aapl_adj_open is inconsistent with open/close adjustment"
                )
        object.__setattr__(self, "session_date", session_date)
        object.__setattr__(self, "aapl_open", raw_open)
        object.__setattr__(self, "aapl_close", raw_close)
        object.__setattr__(self, "aapl_adj_close", adjusted_close)
        object.__setattr__(self, "spy_adj_close", spy)
        object.__setattr__(self, "qqq_adj_close", qqq)
        object.__setattr__(self, "contextual_signal", contextual)
        object.__setattr__(self, "weak_trend_signal", weak)
        object.__setattr__(self, "canonical_union_opportunity", opportunity)
        object.__setattr__(self, "aapl_adj_open", derived)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date,
            "aapl_open": self.aapl_open,
            "aapl_close": self.aapl_close,
            "aapl_adj_close": self.aapl_adj_close,
            "aapl_adj_open": self.aapl_adj_open,
            "spy_adj_close": self.spy_adj_close,
            "qqq_adj_close": self.qqq_adj_close,
            "contextual_signal": self.contextual_signal,
            "weak_trend_signal": self.weak_trend_signal,
            "canonical_union_opportunity": self.canonical_union_opportunity,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketSession":
        expected = {
            "session_date",
            "aapl_open",
            "aapl_close",
            "aapl_adj_close",
            "aapl_adj_open",
            "spy_adj_close",
            "qqq_adj_close",
            "contextual_signal",
            "weak_trend_signal",
            "canonical_union_opportunity",
        }
        _strict_keys(payload, expected, field_name="market-history session")
        return cls(**{name: payload[name] for name in expected})


@dataclass(frozen=True)
class SignalMarketContext:
    """Exact signal-close and lagged closes that define one market state."""

    spy_adj_close: float
    qqq_adj_close: float
    spy_adj_close_10: float
    qqq_adj_close_10: float
    spy_adj_close_20: float
    qqq_adj_close_20: float

    def __post_init__(self) -> None:
        for field_name in (
            "spy_adj_close",
            "qqq_adj_close",
            "spy_adj_close_10",
            "qqq_adj_close_10",
            "spy_adj_close_20",
            "qqq_adj_close_20",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_positive(getattr(self, field_name), field_name=field_name),
            )

    @property
    def market_state(self) -> str:
        fast_on = bool(
            self.spy_adj_close / self.spy_adj_close_10 - 1.0 > 0.0
            and self.qqq_adj_close / self.qqq_adj_close_10 - 1.0 > 0.0
        )
        slow_on = bool(
            self.spy_adj_close / self.spy_adj_close_20 - 1.0 > 0.0
            and self.qqq_adj_close / self.qqq_adj_close_20 - 1.0 > 0.0
        )
        return (
            f"fast_{'on' if fast_on else 'off'}_"
            f"slow_{'on' if slow_on else 'off'}"
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "spy_adj_close": self.spy_adj_close,
            "qqq_adj_close": self.qqq_adj_close,
            "spy_adj_close_10": self.spy_adj_close_10,
            "qqq_adj_close_10": self.qqq_adj_close_10,
            "spy_adj_close_20": self.spy_adj_close_20,
            "qqq_adj_close_20": self.qqq_adj_close_20,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SignalMarketContext":
        expected = {
            "spy_adj_close",
            "qqq_adj_close",
            "spy_adj_close_10",
            "qqq_adj_close_10",
            "spy_adj_close_20",
            "qqq_adj_close_20",
        }
        _strict_keys(payload, expected, field_name="signal market context")
        return cls(**{name: payload[name] for name in expected})


@dataclass
class _PoolState:
    cumulative_rewards: dict[str, float]
    range_energy: float = 0.0
    effective_count: float = 0.0

    @classmethod
    def empty(cls) -> "_PoolState":
        return cls({name: 0.0 for name in EXPERT_NAMES})

    def _validated_components(
        self,
    ) -> tuple[dict[str, float], float, float]:
        _strict_keys(
            self.cumulative_rewards,
            set(EXPERT_NAMES),
            field_name="pool cumulative_rewards",
        )
        rewards = {
            name: _checked_normal_or_zero(
                self.cumulative_rewards[name], field_name=f"pool reward[{name}]"
            )
            for name in EXPERT_NAMES
        }
        range_energy = _finite_nonnegative(
            self.range_energy, field_name="range_energy"
        )
        effective_count = _finite_nonnegative(
            self.effective_count, field_name="effective_count"
        )
        if range_energy == 0.0 and any(value != 0.0 for value in rewards.values()):
            raise ValueError("zero range energy requires zero cumulative rewards")
        if effective_count == 0.0 and (
            range_energy != 0.0 or any(value != 0.0 for value in rewards.values())
        ):
            raise ValueError("empty pool must have zero sufficient state")
        if effective_count > 0.0 and range_energy > 0.0:
            cauchy_bound = _finite_positive(
                math.sqrt(effective_count) * math.sqrt(range_energy),
                field_name="pool Cauchy bound",
            )
            for name, value in rewards.items():
                if abs(value) > cauchy_bound and not math.isclose(
                    abs(value),
                    cauchy_bound,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ):
                    raise ValueError(
                        f"pool reward[{name}] violates the Cauchy bound"
                    )
        return rewards, range_energy, effective_count

    def decay(self, discount: float) -> None:
        rewards, range_energy, effective_count = self._validated_components()
        decayed_rewards = {
            name: _checked_normal_or_zero(
                rewards[name] * discount,
                field_name=f"decayed cumulative reward[{name}]",
            )
            for name in EXPERT_NAMES
        }
        decayed_energy = _finite_nonnegative(
            range_energy * discount, field_name="decayed range_energy"
        )
        decayed_count = _finite_nonnegative(
            effective_count * discount,
            field_name="decayed effective_count",
        )
        candidate = _PoolState(
            decayed_rewards,
            range_energy=decayed_energy,
            effective_count=decayed_count,
        )
        candidate._validated_components()
        self.cumulative_rewards = decayed_rewards
        self.range_energy = decayed_energy
        self.effective_count = decayed_count

    def add(self, rewards: Mapping[str, float], reward_range: float) -> None:
        current_rewards, current_energy, current_count = self._validated_components()
        _strict_keys(rewards, set(EXPERT_NAMES), field_name="lesson rewards")
        reward_range = _finite_nonnegative(
            reward_range, field_name="reward_range"
        )
        lesson_rewards = {
            name: _checked_normal_or_zero(
                rewards[name], field_name=f"reward[{name}]"
            )
            for name in EXPERT_NAMES
        }
        updated_rewards = {
            name: _checked_normal_or_zero(
                current_rewards[name] + lesson_rewards[name],
                field_name=f"updated cumulative reward[{name}]",
            )
            for name in EXPERT_NAMES
        }
        squared_range = _finite_nonnegative(
            reward_range**2, field_name="squared reward_range"
        )
        updated_energy = _finite_nonnegative(
            current_energy + squared_range,
            field_name="updated range_energy",
        )
        updated_count = _finite_nonnegative(
            current_count + 1.0,
            field_name="updated effective_count",
        )
        candidate = _PoolState(
            updated_rewards,
            range_energy=updated_energy,
            effective_count=updated_count,
        )
        candidate._validated_components()
        self.cumulative_rewards = updated_rewards
        self.range_energy = updated_energy
        self.effective_count = updated_count

    def weights(self) -> dict[str, float]:
        rewards, range_energy, _ = self._validated_components()
        if range_energy == 0.0:
            return {name: 1.0 / len(EXPERT_NAMES) for name in EXPERT_NAMES}
        eta = _finite_positive(
            math.sqrt(
                2.0 * math.log(float(len(EXPERT_NAMES))) / range_energy
            ),
            field_name="self-confident eta",
        )
        maximum = max(rewards.values())
        logits = {
            name: _checked_normal_or_zero(
                eta * (rewards[name] - maximum),
                field_name=f"expert logit[{name}]",
            )
            for name in EXPERT_NAMES
        }
        unnormalized = {
            name: _finite_nonnegative(
                math.exp(logits[name]),
                field_name=f"unnormalized expert weight[{name}]",
            )
            for name in EXPERT_NAMES
        }
        denominator = _finite_positive(
            sum(unnormalized.values()), field_name="expert weight denominator"
        )
        weights = {
            name: _finite_nonnegative(
                unnormalized[name] / denominator,
                field_name=f"expert weight[{name}]",
            )
            for name in EXPERT_NAMES
        }
        if not math.isclose(
            sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("expert weights do not sum to one")
        return weights

    def to_dict(self) -> dict[str, Any]:
        rewards, range_energy, effective_count = self._validated_components()
        return {
            "cumulative_rewards": {
                name: float(rewards[name]) for name in EXPERT_NAMES
            },
            "range_energy": float(range_energy),
            "effective_count": float(effective_count),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_PoolState":
        _strict_keys(
            payload,
            {"cumulative_rewards", "range_energy", "effective_count"},
            field_name="pool state",
        )
        raw_rewards = payload["cumulative_rewards"]
        _strict_keys(
            raw_rewards, set(EXPERT_NAMES), field_name="pool cumulative_rewards"
        )
        rewards: dict[str, float] = {}
        for name in EXPERT_NAMES:
            rewards[name] = _checked_normal_or_zero(
                raw_rewards[name], field_name=f"pool reward[{name}]"
            )
        state = cls(
            cumulative_rewards=rewards,
            range_energy=_finite_nonnegative(
                payload["range_energy"], field_name="range_energy"
            ),
            effective_count=_finite_nonnegative(
                payload["effective_count"], field_name="effective_count"
            ),
        )
        state._validated_components()
        return state


def _immutable_expert_advice(value: Mapping[str, Any]) -> Mapping[str, bool]:
    _strict_keys(value, set(EXPERT_NAMES), field_name="pending advice")
    advice = {
        name: _strict_bool(value[name], field_name=f"advice[{name}]")
        for name in EXPERT_NAMES
    }
    if advice["always_long"] or not advice["union_cash"]:
        raise ValueError("pending advice violates fixed expert definitions")
    if not (advice["contextual_only"] or advice["weak_trend_only"]):
        raise ValueError("pending advice requires a contributing fixed expert")
    return MappingProxyType(advice)


@dataclass(frozen=True)
class PendingExpertLesson:
    signal_date: str
    sessions_until_maturity: int
    entry_adjusted_open: float | None
    market_state: str | None
    market_context: SignalMarketContext | None
    expert_cash_advice: Mapping[str, bool]
    learning_eligible: bool

    def __post_init__(self) -> None:
        signal_date = _iso_date(self.signal_date, field_name="signal_date")
        remaining = self.sessions_until_maturity
        if isinstance(remaining, bool) or remaining not in (1, 2):
            raise ValueError("sessions_until_maturity must be one or two")
        entry = self.entry_adjusted_open
        if remaining == 2:
            if entry is not None:
                raise ValueError("new pending lesson cannot know entry open")
        else:
            entry = _finite_positive(entry, field_name="entry_adjusted_open")
        market_state = self.market_state
        if market_state is not None and market_state not in MARKET_STATE_NAMES:
            raise ValueError("pending lesson has invalid market state")
        context = self.market_context
        if context is not None and not isinstance(context, SignalMarketContext):
            raise TypeError("market_context must be SignalMarketContext or None")
        recomputed_state = None if context is None else context.market_state
        if market_state != recomputed_state:
            raise ValueError(
                "pending market state is inconsistent with signal-time context"
            )
        advice = _immutable_expert_advice(self.expert_cash_advice)
        eligible = _strict_bool(
            self.learning_eligible, field_name="learning_eligible"
        )
        expected_eligible = _date_value(signal_date) >= LESSON_MEMORY_START
        if eligible != expected_eligible:
            raise ValueError(
                "pending learning eligibility is inconsistent with signal date"
            )
        object.__setattr__(self, "signal_date", signal_date)
        object.__setattr__(self, "sessions_until_maturity", int(remaining))
        object.__setattr__(self, "entry_adjusted_open", entry)
        object.__setattr__(self, "expert_cash_advice", advice)
        object.__setattr__(self, "learning_eligible", eligible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_date": self.signal_date,
            "sessions_until_maturity": self.sessions_until_maturity,
            "entry_adjusted_open": self.entry_adjusted_open,
            "market_state": self.market_state,
            "market_context": (
                None if self.market_context is None else self.market_context.to_dict()
            ),
            "expert_cash_advice": {
                name: bool(self.expert_cash_advice[name]) for name in EXPERT_NAMES
            },
            "learning_eligible": self.learning_eligible,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PendingExpertLesson":
        expected = {
            "signal_date",
            "sessions_until_maturity",
            "entry_adjusted_open",
            "market_state",
            "market_context",
            "expert_cash_advice",
            "learning_eligible",
        }
        _strict_keys(payload, expected, field_name="pending lesson")
        raw_context = payload["market_context"]
        context = (
            None
            if raw_context is None
            else SignalMarketContext.from_dict(raw_context)
        )
        return cls(
            signal_date=payload["signal_date"],
            sessions_until_maturity=payload["sessions_until_maturity"],
            entry_adjusted_open=payload["entry_adjusted_open"],
            market_state=payload["market_state"],
            market_context=context,
            expert_cash_advice=payload["expert_cash_advice"],
            learning_eligible=payload["learning_eligible"],
        )


@dataclass(frozen=True)
class MaturedExpertLesson:
    signal_date: str
    maturity_date: str
    market_state: str | None
    net_cash_log_edge_10bps: float
    expert_cash_advice: Mapping[str, bool]
    expert_rewards: Mapping[str, float]
    reward_range: float
    admitted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_date": self.signal_date,
            "maturity_date": self.maturity_date,
            "market_state": self.market_state or UNKNOWN_MARKET_STATE,
            "net_cash_log_edge_10bps": self.net_cash_log_edge_10bps,
            "expert_cash_advice": {
                name: bool(self.expert_cash_advice[name]) for name in EXPERT_NAMES
            },
            "expert_rewards": {
                name: float(self.expert_rewards[name]) for name in EXPERT_NAMES
            },
            "reward_range": self.reward_range,
            "admitted": self.admitted,
        }


@dataclass(frozen=True)
class AggregationDecision:
    session_date: str
    adjusted_open: float
    market_state: str
    opportunity: bool
    action: Action
    cash_score: float
    expert_cash_advice: Mapping[str, bool]
    aggregate_expert_weights: Mapping[str, float]
    scale_expert_weights: Mapping[str, Mapping[str, float]]
    scale_state_pooling: Mapping[str, float]
    matured_lessons: tuple[MaturedExpertLesson, ...]
    pending_lesson_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date,
            "adjusted_open": self.adjusted_open,
            "market_state": self.market_state,
            "opportunity": self.opportunity,
            "action": self.action,
            "cash_score": self.cash_score,
            "expert_cash_advice": dict(self.expert_cash_advice),
            "aggregate_expert_weights": dict(self.aggregate_expert_weights),
            "scale_expert_weights": {
                scale: dict(weights)
                for scale, weights in self.scale_expert_weights.items()
            },
            "scale_state_pooling": dict(self.scale_state_pooling),
            "matured_lessons": [lesson.to_dict() for lesson in self.matured_lessons],
            "pending_lesson_count": self.pending_lesson_count,
        }


def resolve_cash_action(cash_score: float, *, opportunity: bool) -> Action:
    """Apply the frozen strict CASH threshold; every tie remains LONG."""

    score = _finite_nonnegative(cash_score, field_name="cash score")
    if score > 1.0:
        raise ValueError("cash score cannot exceed one")
    active = _strict_bool(opportunity, field_name="opportunity")
    return (
        "CASH"
        if active and score > 0.5 + LONG_TIE_TOLERANCE
        else "LONG"
    )


class ContextualExpertAggregator:
    """Sequential four-expert LONG/CASH aggregator with causal t+2 lessons."""

    def __init__(
        self,
        *,
        learning_mode: LearningMode = CAUSAL_ONLINE_MODE,
        frozen_cutoff: DateLike | None = None,
        ablation_mode: AblationMode = FULL_MODE,
    ) -> None:
        if not isinstance(learning_mode, str) or learning_mode not in LEARNING_MODES:
            raise ValueError(f"unsupported learning_mode: {learning_mode!r}")
        if not isinstance(ablation_mode, str) or ablation_mode not in ABLATION_MODES:
            raise ValueError(f"unsupported ablation_mode: {ablation_mode!r}")
        if learning_mode == FROZEN_CUTOFF_MODE:
            if frozen_cutoff is None:
                raise ValueError("frozen_cutoff mode requires frozen_cutoff")
            cutoff = _iso_date(frozen_cutoff, field_name="frozen_cutoff")
        else:
            if frozen_cutoff is not None:
                raise ValueError("causal_online mode must not receive frozen_cutoff")
            cutoff = None
        self.learning_mode = learning_mode
        self.frozen_cutoff = cutoff
        self.ablation_mode = ablation_mode
        self.lesson_memory_start = LESSON_MEMORY_START_ISO
        self._global_pools = {
            scale: _PoolState.empty() for scale in SCALE_NAMES
        }
        self._state_pools = {
            scale: {
                state: _PoolState.empty() for state in MARKET_STATE_NAMES
            }
            for scale in SCALE_NAMES
        }
        self._pending: list[PendingExpertLesson] = []
        self._market_history: list[MarketSession] = []
        self._last_session_date: str | None = None
        self._processed_session_count = 0
        self._matured_lesson_count = 0
        self._admitted_lesson_count = 0

    @property
    def pending_lessons(self) -> tuple[PendingExpertLesson, ...]:
        return tuple(self._pending)

    @property
    def matured_lesson_count(self) -> int:
        return self._matured_lesson_count

    @property
    def admitted_lesson_count(self) -> int:
        return self._admitted_lesson_count

    def _market_context_at(self, position: int) -> SignalMarketContext | None:
        if position < SLOW_LOOKBACK:
            return None
        current = self._market_history[position]
        fast = self._market_history[position - FAST_LOOKBACK]
        slow = self._market_history[position - SLOW_LOOKBACK]
        return SignalMarketContext(
            spy_adj_close=current.spy_adj_close,
            qqq_adj_close=current.qqq_adj_close,
            spy_adj_close_10=fast.spy_adj_close,
            qqq_adj_close_10=fast.qqq_adj_close,
            spy_adj_close_20=slow.spy_adj_close,
            qqq_adj_close_20=slow.qqq_adj_close,
        )

    def _market_context(self) -> SignalMarketContext | None:
        return self._market_context_at(len(self._market_history) - 1)

    def _expected_pending_from_history(self) -> list[PendingExpertLesson]:
        if not self._market_history:
            return []
        last = len(self._market_history) - 1
        expected: list[PendingExpertLesson] = []
        for position in range(max(0, last - 1), last + 1):
            session = self._market_history[position]
            if not session.canonical_union_opportunity:
                continue
            remaining = 2 - (last - position)
            entry = (
                None
                if remaining == 2
                else self._market_history[position + 1].aapl_adj_open
            )
            context = self._market_context_at(position)
            expected.append(
                PendingExpertLesson(
                    signal_date=str(session.session_date),
                    sessions_until_maturity=remaining,
                    entry_adjusted_open=entry,
                    market_state=(
                        None if context is None else context.market_state
                    ),
                    market_context=context,
                    expert_cash_advice={
                        "always_long": False,
                        "union_cash": True,
                        "contextual_only": bool(session.contextual_signal),
                        "weak_trend_only": bool(session.weak_trend_signal),
                    },
                    learning_eligible=(
                        _date_value(str(session.session_date))
                        >= LESSON_MEMORY_START
                    ),
                )
            )
        return expected

    def _validate_pending_consistency(self) -> None:
        expected = [
            lesson.to_dict() for lesson in self._expected_pending_from_history()
        ]
        observed = [lesson.to_dict() for lesson in self._pending]
        if observed != expected:
            raise ValueError(
                "pending signal is absent from the market-history tail or its "
                "record differs from the bounded session replay tail"
            )

    def _admit_label(
        self,
        lesson: PendingExpertLesson,
        *,
        maturity_date: str,
        rewards: Mapping[str, float],
        reward_range: float,
    ) -> bool:
        allowed = lesson.learning_eligible and (
            self.learning_mode == CAUSAL_ONLINE_MODE
            or _date_value(maturity_date) <= _date_value(str(self.frozen_cutoff))
        )
        if not allowed:
            return False
        for scale in SCALE_NAMES:
            discount = SCALE_DISCOUNTS[scale]
            self._global_pools[scale].decay(discount)
            for state_pool in self._state_pools[scale].values():
                state_pool.decay(discount)
            self._global_pools[scale].add(rewards, reward_range)
            if lesson.market_state is not None:
                self._state_pools[scale][lesson.market_state].add(
                    rewards, reward_range
                )
        self._admitted_lesson_count += 1
        return True

    def _advance_pending(
        self, *, session_date: str, adjusted_open: float
    ) -> tuple[MaturedExpertLesson, ...]:
        next_pending: list[PendingExpertLesson] = []
        matured: list[MaturedExpertLesson] = []
        for lesson in self._pending:
            if lesson.sessions_until_maturity == 2:
                next_pending.append(
                    PendingExpertLesson(
                        signal_date=lesson.signal_date,
                        sessions_until_maturity=1,
                        entry_adjusted_open=adjusted_open,
                        market_state=lesson.market_state,
                        market_context=lesson.market_context,
                        expert_cash_advice=dict(lesson.expert_cash_advice),
                        learning_eligible=lesson.learning_eligible,
                    )
                )
                continue
            if lesson.sessions_until_maturity != 1:
                raise RuntimeError("pending lesson has invalid maturity distance")
            if lesson.entry_adjusted_open is None:
                raise RuntimeError("maturing lesson is missing its entry open")
            label = _checked_normal_or_zero(
                math.log(lesson.entry_adjusted_open / adjusted_open)
                + ROUND_TRIP_LOG_FRICTION,
                field_name="matured net cash log edge",
            )
            rewards = {
                name: label if lesson.expert_cash_advice[name] else 0.0
                for name in EXPERT_NAMES
            }
            reward_range = _finite_nonnegative(
                max(rewards.values()) - min(rewards.values()),
                field_name="matured reward_range",
            )
            admitted = self._admit_label(
                lesson,
                maturity_date=session_date,
                rewards=rewards,
                reward_range=reward_range,
            )
            self._matured_lesson_count += 1
            matured.append(
                MaturedExpertLesson(
                    signal_date=lesson.signal_date,
                    maturity_date=session_date,
                    market_state=lesson.market_state,
                    net_cash_log_edge_10bps=label,
                    expert_cash_advice=dict(lesson.expert_cash_advice),
                    expert_rewards=rewards,
                    reward_range=reward_range,
                    admitted=admitted,
                )
            )
        self._pending = next_pending
        return tuple(matured)

    def _weights_for_scale(
        self, scale: str, market_state: str | None
    ) -> tuple[dict[str, float], float]:
        global_weights = self._global_pools[scale].weights()
        if self.ablation_mode == GLOBAL_ONLY_MODE or market_state is None:
            return global_weights, 0.0
        state_pool = self._state_pools[scale][market_state]
        rho_denominator = _finite_positive(
            state_pool.effective_count + STATE_PRIOR_STRENGTH,
            field_name="state pooling denominator",
        )
        rho = _finite_nonnegative(
            state_pool.effective_count / rho_denominator,
            field_name="state pooling rho",
        )
        if rho >= 1.0:
            raise ValueError("state pooling rho must be below one")
        state_weights = state_pool.weights()
        pooled = {
            name: _finite_nonnegative(
                (1.0 - rho) * global_weights[name]
                + rho * state_weights[name],
                field_name=f"pooled expert weight[{name}]",
            )
            for name in EXPERT_NAMES
        }
        if not math.isclose(sum(pooled.values()), 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("pooled expert weights do not sum to one")
        return pooled, rho

    def _decision_weights(
        self, market_state: str | None
    ) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
        scales = (
            (LIFETIME_SCALE,)
            if self.ablation_mode == LIFETIME_ONLY_MODE
            else SCALE_NAMES
        )
        per_scale: dict[str, dict[str, float]] = {}
        rhos: dict[str, float] = {}
        for scale in scales:
            weights, rho = self._weights_for_scale(scale, market_state)
            per_scale[scale] = weights
            rhos[scale] = rho
        aggregate = {
            name: _finite_nonnegative(
                sum(per_scale[scale][name] for scale in scales) / len(scales),
                field_name=f"scale-average expert weight[{name}]",
            )
            for name in EXPERT_NAMES
        }
        if not math.isclose(
            sum(aggregate.values()), 1.0, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError("scale-average expert weights do not sum to one")
        return per_scale, aggregate, rhos

    def process_session(self, raw_session: MarketSession) -> AggregationDecision:
        """Advance one strict session and return its post-update decision."""

        session = (
            raw_session
            if isinstance(raw_session, MarketSession)
            else MarketSession(**dict(raw_session))
        )
        session_date = str(session.session_date)
        if self._last_session_date is not None and _date_value(
            session_date
        ) <= _date_value(self._last_session_date):
            raise ValueError("sessions must be supplied in strictly increasing order")

        adjusted_open = float(session.aapl_adj_open)
        matured = self._advance_pending(
            session_date=session_date, adjusted_open=adjusted_open
        )
        self._market_history.append(session)
        self._market_history = self._market_history[-(SLOW_LOOKBACK + 2) :]
        market_context = self._market_context()
        market_state = (
            None if market_context is None else market_context.market_state
        )

        opportunity = bool(session.canonical_union_opportunity)
        advice = {
            "always_long": False,
            "union_cash": opportunity,
            "contextual_only": bool(opportunity and session.contextual_signal),
            "weak_trend_only": bool(opportunity and session.weak_trend_signal),
        }
        per_scale, aggregate, rhos = self._decision_weights(market_state)
        cash_score = _finite_nonnegative(
            sum(aggregate[name] for name in EXPERT_NAMES if advice[name]),
            field_name="cash score",
        )
        action = resolve_cash_action(cash_score, opportunity=opportunity)

        if opportunity:
            self._pending.append(
                PendingExpertLesson(
                    signal_date=session_date,
                    sessions_until_maturity=2,
                    entry_adjusted_open=None,
                    market_state=market_state,
                    market_context=market_context,
                    expert_cash_advice=dict(advice),
                    learning_eligible=(
                        _date_value(session_date)
                        >= LESSON_MEMORY_START
                    ),
                )
            )
        self._last_session_date = session_date
        self._processed_session_count += 1
        self._validate_pending_consistency()
        return AggregationDecision(
            session_date=session_date,
            adjusted_open=adjusted_open,
            market_state=market_state or UNKNOWN_MARKET_STATE,
            opportunity=opportunity,
            action=action,
            cash_score=cash_score,
            expert_cash_advice=advice,
            aggregate_expert_weights=aggregate,
            scale_expert_weights=per_scale,
            scale_state_pooling=rhos,
            matured_lessons=matured,
            pending_lesson_count=len(self._pending),
        )

    def _constants_payload(self) -> dict[str, Any]:
        return {
            "expert_order": list(EXPERT_NAMES),
            "market_state_order": list(MARKET_STATE_NAMES),
            "fast_lookback": FAST_LOOKBACK,
            "slow_lookback": SLOW_LOOKBACK,
            "scale_order": list(SCALE_NAMES),
            "scale_half_lives": {
                name: SCALE_HALF_LIVES[name] for name in SCALE_NAMES
            },
            "per_side_friction": PER_SIDE_FRICTION,
            "round_trip_log_friction": ROUND_TRIP_LOG_FRICTION,
            "state_prior_strength": STATE_PRIOR_STRENGTH,
            "long_tie_tolerance": LONG_TIE_TOLERANCE,
            "lesson_maturity_sessions": 2,
            "lesson_memory_start": LESSON_MEMORY_START_ISO,
        }

    def _validate_state_consistency(self) -> None:
        admitted = self._admitted_lesson_count
        for scale in SCALE_NAMES:
            discount = SCALE_DISCOUNTS[scale]
            # Reproduce the exact update recursion rather than comparing an
            # iterated checkpoint value with an algebraically equivalent
            # closed form that rounds differently.
            expected_count = 0.0
            for _ in range(admitted):
                expected_count = expected_count * discount + 1.0
            global_pool = self._global_pools[scale]
            global_rewards, global_energy, global_count = (
                global_pool._validated_components()
            )
            state_pools = self._state_pools[scale]
            validated_states = {
                state_name: state_pool._validated_components()
                for state_name, state_pool in state_pools.items()
            }
            if admitted == 0:
                cold_components = [
                    (global_rewards, global_energy, global_count),
                    *validated_states.values(),
                ]
                if any(
                    count != 0.0
                    or energy != 0.0
                    or any(value != 0.0 for value in rewards.values())
                    for rewards, energy, count in cold_components
                ):
                    raise ValueError(
                        f"cold state[{scale}] must be exactly zero"
                    )
            if global_count != expected_count:
                raise ValueError(
                    f"global effective_count[{scale}] is inconsistent with "
                    "admitted_lesson_count"
                )
            for state_name, (_, state_energy, state_count) in validated_states.items():
                count_margin = _rounding_margin(state_count, global_count)
                if state_count > global_count + count_margin:
                    raise ValueError(
                        f"state effective_count[{scale}][{state_name}] exceeds global"
                    )
                energy_margin = _rounding_margin(state_energy, global_energy)
                if state_energy > global_energy + energy_margin:
                    raise ValueError(
                        f"state range_energy[{scale}][{state_name}] exceeds global"
                    )
            state_count_sum = sum(
                components[2] for components in validated_states.values()
            )
            if state_count_sum > global_count + _rounding_margin(
                state_count_sum, global_count
            ):
                raise ValueError(
                    f"summed state effective_count[{scale}] exceeds global"
                )
            state_energy_sum = sum(
                components[1] for components in validated_states.values()
            )
            if state_energy_sum > global_energy + _rounding_margin(
                state_energy_sum, global_energy
            ):
                raise ValueError(
                    f"summed state range_energy[{scale}] exceeds global"
                )

        expected_history_length = min(
            self._processed_session_count, SLOW_LOOKBACK + 2
        )
        if len(self._market_history) != expected_history_length:
            raise ValueError(
                "bounded market-history length is inconsistent with processed sessions"
            )
        if self._matured_lesson_count > max(
            0, self._processed_session_count - 2
        ):
            raise ValueError("matured lesson count exceeds processed session capacity")
        if self._market_history:
            if self._last_session_date != str(self._market_history[-1].session_date):
                raise ValueError("market-history tail does not end at last_session_date")
        elif (
            self._last_session_date is not None
            or self._matured_lesson_count
            or self._processed_session_count
        ):
            raise ValueError("nonempty model state requires market history")
        self._validate_pending_consistency()

    def state_dict(self) -> dict[str, Any]:
        """Return model state without runtime learning/ablation controls."""

        self._validate_state_consistency()
        return {
            "last_session_date": self._last_session_date,
            "processed_session_count": self._processed_session_count,
            "matured_lesson_count": self._matured_lesson_count,
            "admitted_lesson_count": self._admitted_lesson_count,
            "market_history": [session.to_dict() for session in self._market_history],
            "pending_lessons": [lesson.to_dict() for lesson in self._pending],
            "global_pools": {
                scale: self._global_pools[scale].to_dict()
                for scale in SCALE_NAMES
            },
            "state_pools": {
                scale: {
                    state: self._state_pools[scale][state].to_dict()
                    for state in MARKET_STATE_NAMES
                }
                for scale in SCALE_NAMES
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "serialization_version": SERIALIZATION_VERSION,
            "method": MODEL_METHOD,
            "constants": self._constants_payload(),
            "runtime": {
                "learning_mode": self.learning_mode,
                "frozen_cutoff": self.frozen_cutoff,
                "ablation_mode": self.ablation_mode,
            },
            "state": self.state_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        learning_mode: LearningMode | None = None,
        frozen_cutoff: DateLike | None = None,
        ablation_mode: AblationMode | None = None,
    ) -> "ContextualExpertAggregator":
        _strict_keys(
            payload,
            {"serialization_version", "method", "constants", "runtime", "state"},
            field_name="serialized aggregator",
        )
        if payload["serialization_version"] != SERIALIZATION_VERSION:
            raise ValueError("unsupported serialization_version")
        if payload["method"] != MODEL_METHOD:
            raise ValueError("serialized method is incompatible")
        expected_constants = cls()._constants_payload()
        if payload["constants"] != expected_constants:
            raise ValueError("serialized constants are incompatible")
        runtime = payload["runtime"]
        _strict_keys(
            runtime,
            {"learning_mode", "frozen_cutoff", "ablation_mode"},
            field_name="serialized runtime",
        )
        # Validate the serialized controls even when the caller requests an
        # override.  An override must never make a malformed checkpoint load.
        cls(
            learning_mode=runtime["learning_mode"],
            frozen_cutoff=runtime["frozen_cutoff"],
            ablation_mode=runtime["ablation_mode"],
        )
        selected_learning_mode = (
            runtime["learning_mode"] if learning_mode is None else learning_mode
        )
        selected_ablation = (
            runtime["ablation_mode"] if ablation_mode is None else ablation_mode
        )
        selected_cutoff: DateLike | None
        if selected_learning_mode == CAUSAL_ONLINE_MODE:
            selected_cutoff = frozen_cutoff
        else:
            selected_cutoff = (
                frozen_cutoff
                if frozen_cutoff is not None
                else (
                    runtime["frozen_cutoff"]
                    if runtime["learning_mode"] == FROZEN_CUTOFF_MODE
                    else None
                )
            )
        state = payload["state"]
        _strict_keys(
            state,
            {
                "last_session_date",
                "processed_session_count",
                "matured_lesson_count",
                "admitted_lesson_count",
                "market_history",
                "pending_lessons",
                "global_pools",
                "state_pools",
            },
            field_name="serialized state",
        )
        model = cls(
            learning_mode=selected_learning_mode,
            frozen_cutoff=selected_cutoff,
            ablation_mode=selected_ablation,
        )
        last = state["last_session_date"]
        model._last_session_date = (
            None if last is None else _iso_date(last, field_name="last_session_date")
        )
        for count_name in (
            "processed_session_count",
            "matured_lesson_count",
            "admitted_lesson_count",
        ):
            value = state[count_name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{count_name} must be a non-negative integer")
        model._processed_session_count = state["processed_session_count"]
        model._matured_lesson_count = state["matured_lesson_count"]
        model._admitted_lesson_count = state["admitted_lesson_count"]
        if model._admitted_lesson_count > model._matured_lesson_count:
            raise ValueError("admitted lesson count cannot exceed matured count")

        history = state["market_history"]
        if not isinstance(history, list) or len(history) > SLOW_LOOKBACK + 2:
            raise ValueError("market_history must be a bounded list")
        previous: str | None = None
        parsed_history: list[MarketSession] = []
        for row in history:
            session = MarketSession.from_dict(row)
            row_date = str(session.session_date)
            if previous is not None and _date_value(row_date) <= _date_value(previous):
                raise ValueError("market history must be strictly chronological")
            previous = row_date
            parsed_history.append(session)
        if (
            parsed_history
            and model._last_session_date != str(parsed_history[-1].session_date)
        ):
            raise ValueError("market history does not end at last_session_date")
        if not parsed_history and model._last_session_date is not None:
            raise ValueError("last_session_date requires market history")
        model._market_history = parsed_history

        raw_global = state["global_pools"]
        _strict_keys(raw_global, set(SCALE_NAMES), field_name="global pools")
        raw_states = state["state_pools"]
        _strict_keys(raw_states, set(SCALE_NAMES), field_name="state pools")
        model._global_pools = {
            scale: _PoolState.from_dict(raw_global[scale]) for scale in SCALE_NAMES
        }
        model._state_pools = {}
        for scale in SCALE_NAMES:
            _strict_keys(
                raw_states[scale],
                set(MARKET_STATE_NAMES),
                field_name=f"state pools[{scale}]",
            )
            model._state_pools[scale] = {
                name: _PoolState.from_dict(raw_states[scale][name])
                for name in MARKET_STATE_NAMES
            }

        raw_pending = state["pending_lessons"]
        if not isinstance(raw_pending, list):
            raise ValueError("pending_lessons must be a list")
        pending: list[PendingExpertLesson] = []
        previous_signal: str | None = None
        for row in raw_pending:
            lesson = PendingExpertLesson.from_dict(row)
            if previous_signal is not None and _date_value(
                lesson.signal_date
            ) <= _date_value(previous_signal):
                raise ValueError("pending lessons must be strictly chronological")
            previous_signal = lesson.signal_date
            pending.append(lesson)
        model._pending = pending
        model._validate_state_consistency()
        return model

    def fork(
        self,
        *,
        learning_mode: LearningMode | None = None,
        frozen_cutoff: DateLike | None = None,
        ablation_mode: AblationMode | None = None,
    ) -> "ContextualExpertAggregator":
        """Clone the exact checkpoint with different evaluation controls."""

        return self.from_dict(
            self.to_dict(),
            learning_mode=learning_mode,
            frozen_cutoff=frozen_cutoff,
            ablation_mode=ablation_mode,
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
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
    def load(
        cls,
        path: str | os.PathLike[str],
        **overrides: Any,
    ) -> "ContextualExpertAggregator":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("serialized aggregator must be a JSON object")
        return cls.from_dict(payload, **overrides)


__all__ = [
    "ABLATION_MODES",
    "AggregationDecision",
    "CAUSAL_ONLINE_MODE",
    "ContextualExpertAggregator",
    "EXPERT_NAMES",
    "FAST_LOOKBACK",
    "FAST_SCALE",
    "FROZEN_CUTOFF_MODE",
    "FULL_MODE",
    "GLOBAL_ONLY_MODE",
    "LIFETIME_ONLY_MODE",
    "LIFETIME_SCALE",
    "LESSON_MEMORY_START_ISO",
    "LONG_TIE_TOLERANCE",
    "MARKET_STATE_NAMES",
    "MEDIUM_SCALE",
    "MODEL_METHOD",
    "MarketSession",
    "MaturedExpertLesson",
    "PendingExpertLesson",
    "ROUND_TRIP_LOG_FRICTION",
    "SCALE_DISCOUNTS",
    "SCALE_NAMES",
    "SERIALIZATION_VERSION",
    "SLOW_LOOKBACK",
    "STATE_PRIOR_STRENGTH",
    "SignalMarketContext",
    "UNKNOWN_MARKET_STATE",
    "derive_adjusted_open",
    "resolve_cash_action",
]
