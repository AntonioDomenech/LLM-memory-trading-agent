"""Causal two-regime selector for the fixed AAPL exhaustion union.

This module owns only the point-in-time model and its continuous shadow
replay.  It does not acquire data, score a historical stage, or construct an
account-reset boundary.  The sealed experiment runner derives the single
2005 account union separately while this core keeps the action-independent
canonical union lesson stream continuous from 2000 onward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .chronological_exhaustion_expert import (
    CAUSAL_ONLINE_MODE,
    FROZEN_CUTOFF_MODE,
    LEARNING_MODES,
    LESSON_MEMORY_START,
    ROUND_TRIP_LOG_FRICTION,
    _canonical_market_frame,
    build_fixed_expert_signals,
)


RISK_ON_REGIME = "risk_on"
NOT_RISK_ON_REGIME = "not_risk_on"
REGIME_NAMES = (RISK_ON_REGIME, NOT_RISK_ON_REGIME)
REGIME_FEATURE_COLUMNS = ("spy_return_20", "qqq_return_20", "risk_on")

REGIME_LOOKBACK = 20
LESSON_DISCOUNT = 0.995
MIN_EFFECTIVE_LESSONS = 12.0
POSITIVE_MEAN_THRESHOLD = 0.001
NEGATIVE_MEAN_THRESHOLD = -0.001
RISK_ON_STRUCTURAL_DEFAULT_CASH = False
NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH = True

STATE_SCHEMA_VERSION = "binary-regime-union-selector-state-v1"
PENDING_SCHEMA_VERSION = "binary-regime-union-selector-pending-v1"

_N_EFF_ABSOLUTE_TOLERANCE = 5e-12
_CAUCHY_RELATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class EwRegimeState:
    """Immutable exponentially weighted sufficient state for one regime."""

    n_raw: int
    n_eff: float
    weighted_label_sum: float
    weighted_squared_label_sum: float
    cash_selected: bool

    @property
    def mean(self) -> float:
        if self.n_eff <= 0.0:
            return math.nan
        return float(self.weighted_label_sum / self.n_eff)

    @property
    def ready(self) -> bool:
        return bool(self.n_eff >= MIN_EFFECTIVE_LESSONS)


@dataclass(frozen=True)
class PendingRegimeLesson:
    """A canonical shadow opportunity whose t+2 open is not yet known."""

    signal_close: pd.Timestamp
    sessions_until_maturity: int
    risk_on: bool
    entry_adjusted_open: float | None
    learning_eligible: bool


@dataclass(frozen=True)
class BinaryRegimeSelectorReplay:
    forecast: pd.DataFrame
    final_states: Mapping[str, EwRegimeState]
    pending_lessons: tuple[PendingRegimeLesson, ...]


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    result = int(numeric)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _state_numeric_values(state: EwRegimeState) -> tuple[int, float, float, float]:
    if not isinstance(state, EwRegimeState):
        raise TypeError("state must be an EwRegimeState")
    n_raw = _integer(state.n_raw, name="n_raw")
    n_eff = float(state.n_eff)
    weighted_sum = float(state.weighted_label_sum)
    weighted_squared_sum = float(state.weighted_squared_label_sum)
    _strict_bool(state.cash_selected, name="cash_selected")
    if not math.isfinite(n_eff) or n_eff < 0.0:
        raise ValueError("n_eff must be finite and non-negative")
    expected_n_eff = (1.0 - LESSON_DISCOUNT**n_raw) / (
        1.0 - LESSON_DISCOUNT
    )
    if not math.isclose(
        n_eff,
        expected_n_eff,
        rel_tol=0.0,
        abs_tol=_N_EFF_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("n_eff is inconsistent with n_raw and lesson discount")
    if not math.isfinite(weighted_sum):
        raise ValueError("weighted_label_sum must be finite")
    if not math.isfinite(weighted_squared_sum) or weighted_squared_sum < 0.0:
        raise ValueError(
            "weighted_squared_label_sum must be finite and non-negative"
        )
    if n_raw == 0:
        if n_eff != 0.0 or weighted_sum != 0.0 or weighted_squared_sum != 0.0:
            raise ValueError("an empty regime state must have zero sufficient state")
    elif n_eff <= 0.0:
        raise ValueError("a non-empty regime state must have positive n_eff")
    cauchy_bound = math.sqrt(n_eff) * math.sqrt(weighted_squared_sum)
    if abs(weighted_sum) > cauchy_bound and not math.isclose(
        abs(weighted_sum),
        cauchy_bound,
        rel_tol=_CAUCHY_RELATIVE_TOLERANCE,
        abs_tol=0.0,
    ):
        raise ValueError(
            "weighted sufficient state violates S^2 <= n_eff * Q"
        )
    return n_raw, n_eff, weighted_sum, weighted_squared_sum


def _validate_regime_state(
    state: EwRegimeState,
    *,
    structural_default_cash: bool,
) -> EwRegimeState:
    _state_numeric_values(state)
    default_cash = _strict_bool(
        structural_default_cash, name="structural_default_cash"
    )
    selected = _strict_bool(state.cash_selected, name="cash_selected")
    if not state.ready:
        if selected != default_cash:
            raise ValueError("an unready regime must retain its structural default")
        return state
    mean = state.mean
    if not math.isfinite(mean):
        raise ValueError("a ready regime must have a finite mean")
    if mean > POSITIVE_MEAN_THRESHOLD and not selected:
        raise ValueError("positive ready regime mean requires the CASH latch")
    if mean < NEGATIVE_MEAN_THRESHOLD and selected:
        raise ValueError("negative ready regime mean requires the LONG latch")
    return state


def initialize_regime_state(*, structural_default_cash: bool) -> EwRegimeState:
    default_cash = _strict_bool(
        structural_default_cash, name="structural_default_cash"
    )
    return EwRegimeState(
        n_raw=0,
        n_eff=0.0,
        weighted_label_sum=0.0,
        weighted_squared_label_sum=0.0,
        cash_selected=default_cash,
    )


def initialize_regime_states() -> dict[str, EwRegimeState]:
    return {
        RISK_ON_REGIME: initialize_regime_state(
            structural_default_cash=RISK_ON_STRUCTURAL_DEFAULT_CASH
        ),
        NOT_RISK_ON_REGIME: initialize_regime_state(
            structural_default_cash=NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH
        ),
    }


def resolve_regime_cash_selection(
    state: EwRegimeState,
    *,
    structural_default_cash: bool,
) -> bool:
    """Apply readiness, strict thresholds, and the frozen hysteresis rule."""

    _state_numeric_values(state)
    default_cash = _strict_bool(
        structural_default_cash, name="structural_default_cash"
    )
    previous = _strict_bool(state.cash_selected, name="cash_selected")
    if state.n_eff < MIN_EFFECTIVE_LESSONS:
        return default_cash
    mean = state.mean
    if not math.isfinite(mean):
        raise ValueError("a ready regime must have a finite mean")
    if mean > POSITIVE_MEAN_THRESHOLD:
        return True
    if mean < NEGATIVE_MEAN_THRESHOLD:
        return False
    return previous


def update_ew_regime_state(
    state: EwRegimeState,
    label: float,
    *,
    structural_default_cash: bool,
) -> EwRegimeState:
    """Admit one matured label to one regime and update its latch."""

    _state_numeric_values(state)
    default_cash = _strict_bool(
        structural_default_cash, name="structural_default_cash"
    )
    value = float(label)
    if not math.isfinite(value):
        raise ValueError("label must be finite")
    provisional = EwRegimeState(
        n_raw=int(state.n_raw) + 1,
        n_eff=LESSON_DISCOUNT * float(state.n_eff) + 1.0,
        weighted_label_sum=(
            LESSON_DISCOUNT * float(state.weighted_label_sum) + value
        ),
        weighted_squared_label_sum=(
            LESSON_DISCOUNT * float(state.weighted_squared_label_sum)
            + value**2
        ),
        cash_selected=bool(state.cash_selected),
    )
    selected = resolve_regime_cash_selection(
        provisional, structural_default_cash=default_cash
    )
    updated = EwRegimeState(
        n_raw=provisional.n_raw,
        n_eff=provisional.n_eff,
        weighted_label_sum=provisional.weighted_label_sum,
        weighted_squared_label_sum=provisional.weighted_squared_label_sum,
        cash_selected=selected,
    )
    return _validate_regime_state(
        updated, structural_default_cash=default_cash
    )


def serialize_ew_regime_state(state: EwRegimeState) -> dict[str, Any]:
    n_raw, n_eff, weighted_sum, weighted_squared_sum = _state_numeric_values(
        state
    )
    return {
        "n_raw": n_raw,
        "n_eff": n_eff,
        "weighted_label_sum": weighted_sum,
        "weighted_squared_label_sum": weighted_squared_sum,
        "cash_selected": _strict_bool(
            state.cash_selected, name="cash_selected"
        ),
    }


def restore_ew_regime_state(
    payload: Mapping[str, Any],
    *,
    structural_default_cash: bool,
) -> EwRegimeState:
    if not isinstance(payload, Mapping):
        raise TypeError("state payload must be a mapping")
    required = {
        "n_raw",
        "n_eff",
        "weighted_label_sum",
        "weighted_squared_label_sum",
        "cash_selected",
    }
    if set(payload) != required:
        raise ValueError("state payload has missing or unexpected fields")
    try:
        state = EwRegimeState(
            n_raw=_integer(payload["n_raw"], name="n_raw"),
            n_eff=float(payload["n_eff"]),
            weighted_label_sum=float(payload["weighted_label_sum"]),
            weighted_squared_label_sum=float(
                payload["weighted_squared_label_sum"]
            ),
            cash_selected=_strict_bool(
                payload["cash_selected"], name="cash_selected"
            ),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("state payload contains invalid values") from exc
    return _validate_regime_state(
        state, structural_default_cash=structural_default_cash
    )


def _strict_regime_state_mapping(
    states: Mapping[str, EwRegimeState],
) -> dict[str, EwRegimeState]:
    if not isinstance(states, Mapping):
        raise TypeError("states must be a mapping")
    if set(states) != set(REGIME_NAMES):
        raise ValueError("states must contain exactly the two frozen regimes")
    risk_on = _validate_regime_state(
        states[RISK_ON_REGIME],
        structural_default_cash=RISK_ON_STRUCTURAL_DEFAULT_CASH,
    )
    not_risk_on = _validate_regime_state(
        states[NOT_RISK_ON_REGIME],
        structural_default_cash=NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH,
    )
    return {RISK_ON_REGIME: risk_on, NOT_RISK_ON_REGIME: not_risk_on}


def serialize_regime_states(
    states: Mapping[str, EwRegimeState],
) -> dict[str, Any]:
    value = _strict_regime_state_mapping(states)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "regime_order": list(REGIME_NAMES),
        "regime_feature_order": list(REGIME_FEATURE_COLUMNS),
        "lesson_discount": LESSON_DISCOUNT,
        "minimum_effective_lessons": MIN_EFFECTIVE_LESSONS,
        "positive_mean_threshold": POSITIVE_MEAN_THRESHOLD,
        "negative_mean_threshold": NEGATIVE_MEAN_THRESHOLD,
        "structural_default_cash": {
            RISK_ON_REGIME: RISK_ON_STRUCTURAL_DEFAULT_CASH,
            NOT_RISK_ON_REGIME: NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH,
        },
        "states": {
            regime: serialize_ew_regime_state(value[regime])
            for regime in REGIME_NAMES
        },
    }


def restore_regime_states(
    payload: Mapping[str, Any],
) -> dict[str, EwRegimeState]:
    if not isinstance(payload, Mapping):
        raise TypeError("regime-state payload must be a mapping")
    required = {
        "schema_version",
        "regime_order",
        "regime_feature_order",
        "lesson_discount",
        "minimum_effective_lessons",
        "positive_mean_threshold",
        "negative_mean_threshold",
        "structural_default_cash",
        "states",
    }
    if set(payload) != required:
        raise ValueError(
            "regime-state payload has missing or unexpected fields"
        )
    expected_constants = {
        "schema_version": STATE_SCHEMA_VERSION,
        "regime_order": list(REGIME_NAMES),
        "regime_feature_order": list(REGIME_FEATURE_COLUMNS),
        "lesson_discount": LESSON_DISCOUNT,
        "minimum_effective_lessons": MIN_EFFECTIVE_LESSONS,
        "positive_mean_threshold": POSITIVE_MEAN_THRESHOLD,
        "negative_mean_threshold": NEGATIVE_MEAN_THRESHOLD,
        "structural_default_cash": {
            RISK_ON_REGIME: RISK_ON_STRUCTURAL_DEFAULT_CASH,
            NOT_RISK_ON_REGIME: NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH,
        },
    }
    if any(payload[name] != expected for name, expected in expected_constants.items()):
        raise ValueError("regime-state payload constants are incompatible")
    raw_states = payload["states"]
    if not isinstance(raw_states, Mapping) or set(raw_states) != set(REGIME_NAMES):
        raise ValueError("regime-state payload has incompatible regimes")
    return {
        RISK_ON_REGIME: restore_ew_regime_state(
            raw_states[RISK_ON_REGIME],
            structural_default_cash=RISK_ON_STRUCTURAL_DEFAULT_CASH,
        ),
        NOT_RISK_ON_REGIME: restore_ew_regime_state(
            raw_states[NOT_RISK_ON_REGIME],
            structural_default_cash=NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH,
        ),
    }


def _validated_pending_lesson(
    lesson: PendingRegimeLesson,
) -> PendingRegimeLesson:
    if not isinstance(lesson, PendingRegimeLesson):
        raise TypeError("pending lessons must contain PendingRegimeLesson values")
    signal_close = pd.Timestamp(lesson.signal_close)
    if pd.isna(signal_close):
        raise ValueError("pending signal_close must be a valid timestamp")
    if signal_close.tzinfo is not None:
        signal_close = signal_close.tz_localize(None)
    remaining = _integer(
        lesson.sessions_until_maturity,
        name="sessions_until_maturity",
        minimum=1,
    )
    if remaining > 2:
        raise ValueError("sessions_until_maturity cannot exceed two")
    risk_on = _strict_bool(lesson.risk_on, name="risk_on")
    eligible = _strict_bool(
        lesson.learning_eligible, name="learning_eligible"
    )
    entry = lesson.entry_adjusted_open
    if entry is not None:
        entry = float(entry)
        if not math.isfinite(entry) or entry <= 0.0:
            raise ValueError(
                "pending entry_adjusted_open must be positive and finite"
            )
    if remaining == 1 and entry is None:
        raise ValueError("a one-session pending lesson must know its entry open")
    if remaining == 2 and entry is not None:
        raise ValueError("a two-session pending lesson cannot know its entry open")
    return PendingRegimeLesson(
        signal_close=signal_close,
        sessions_until_maturity=remaining,
        risk_on=risk_on,
        entry_adjusted_open=entry,
        learning_eligible=eligible,
    )


def serialize_pending_regime_lessons(
    pending_lessons: Sequence[PendingRegimeLesson],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    previous: pd.Timestamp | None = None
    for raw_lesson in pending_lessons:
        lesson = _validated_pending_lesson(raw_lesson)
        if previous is not None and lesson.signal_close <= previous:
            raise ValueError("pending lessons must be strictly chronological")
        previous = lesson.signal_close
        rows.append(
            {
                "signal_close": lesson.signal_close.isoformat(),
                "sessions_until_maturity": lesson.sessions_until_maturity,
                "risk_on": lesson.risk_on,
                "entry_adjusted_open": lesson.entry_adjusted_open,
                "learning_eligible": lesson.learning_eligible,
            }
        )
    return {
        "schema_version": PENDING_SCHEMA_VERSION,
        "regime_order": list(REGIME_NAMES),
        "lessons": rows,
    }


def restore_pending_regime_lessons(
    payload: Mapping[str, Any],
) -> tuple[PendingRegimeLesson, ...]:
    if not isinstance(payload, Mapping):
        raise TypeError("pending payload must be a mapping")
    if set(payload) != {"schema_version", "regime_order", "lessons"}:
        raise ValueError("pending payload has missing or unexpected fields")
    if payload["schema_version"] != PENDING_SCHEMA_VERSION or payload[
        "regime_order"
    ] != list(REGIME_NAMES):
        raise ValueError("pending payload contract is incompatible")
    rows = payload["lessons"]
    if not isinstance(rows, list):
        raise ValueError("pending payload lessons must be a list")
    required = {
        "signal_close",
        "sessions_until_maturity",
        "risk_on",
        "entry_adjusted_open",
        "learning_eligible",
    }
    lessons: list[PendingRegimeLesson] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("pending lesson has missing or unexpected fields")
        try:
            lesson = PendingRegimeLesson(
                signal_close=pd.Timestamp(row["signal_close"]),
                sessions_until_maturity=_integer(
                    row["sessions_until_maturity"],
                    name="sessions_until_maturity",
                    minimum=1,
                ),
                risk_on=_strict_bool(row["risk_on"], name="risk_on"),
                entry_adjusted_open=(
                    None
                    if row["entry_adjusted_open"] is None
                    else float(row["entry_adjusted_open"])
                ),
                learning_eligible=_strict_bool(
                    row["learning_eligible"], name="learning_eligible"
                ),
            )
            lessons.append(_validated_pending_lesson(lesson))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pending lesson contains invalid values") from exc
    # Reuse the serializer for ordering and complete cross-row validation.
    serialize_pending_regime_lessons(lessons)
    return tuple(lessons)


def _validate_learning_contract(
    learning_mode: str,
    frozen_cutoff: str | pd.Timestamp | None,
) -> pd.Timestamp | None:
    if learning_mode not in LEARNING_MODES:
        raise ValueError(
            f"unsupported learning_mode {learning_mode!r}; expected one of "
            f"{sorted(LEARNING_MODES)}"
        )
    if learning_mode == FROZEN_CUTOFF_MODE:
        if frozen_cutoff is None:
            raise ValueError(
                "frozen_cutoff mode requires an inclusive frozen_cutoff"
            )
        cutoff = pd.Timestamp(frozen_cutoff)
        if pd.isna(cutoff):
            raise ValueError("frozen_cutoff must be a valid timestamp")
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        return cutoff
    if frozen_cutoff is not None:
        raise ValueError("causal_online mode must not receive a frozen_cutoff")
    return None


def build_binary_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the exact completed-close 20-session binary regime."""

    data = _canonical_market_frame(frame)
    spy_return = data["spy_adj_close"].pct_change(
        REGIME_LOOKBACK, fill_method=None
    )
    qqq_return = data["qqq_adj_close"].pct_change(
        REGIME_LOOKBACK, fill_method=None
    )
    ready = (
        spy_return.notna()
        & qqq_return.notna()
        & np.isfinite(spy_return)
        & np.isfinite(qqq_return)
    ).astype(bool)
    risk_on = (ready & (spy_return > 0.0) & (qqq_return > 0.0)).astype(bool)
    signals = build_fixed_expert_signals(data)
    candidates = signals["unfiltered_union_candidate_signal"].astype(bool)
    if bool((candidates & ~ready).any()):
        raise ValueError(
            "a raw union candidate has a missing or nonfinite regime return"
        )
    result = pd.DataFrame(
        {
            "spy_return_20": spy_return,
            "qqq_return_20": qqq_return,
            "risk_regime_ready": ready,
            "risk_on": risk_on,
        },
        index=data.index,
    )
    result.attrs.update(
        {
            "regime_feature_order": list(REGIME_FEATURE_COLUMNS),
            "risk_on_definition": (
                "spy adjusted-close 20-session return > 0 AND "
                "qqq adjusted-close 20-session return > 0"
            ),
            "regime_signal_time": "completed close t",
        }
    )
    return result


def _diagnostic_arrays(length: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "shadow_matures_on_close": np.full(
            length, np.datetime64("NaT"), dtype="datetime64[ns]"
        ),
        "shadow_matured_now": np.zeros(length, dtype=bool),
        "shadow_signal_close": np.full(
            length, np.datetime64("NaT"), dtype="datetime64[ns]"
        ),
        "shadow_opportunity_risk_on": np.zeros(length, dtype=bool),
        "shadow_matured_signal_risk_on": np.zeros(length, dtype=bool),
        "shadow_label_10bps": np.full(length, np.nan, dtype=float),
        "shadow_lesson_added_now": np.zeros(length, dtype=bool),
        "shadow_pending": np.zeros(length, dtype=bool),
        "selector_regime_n_eff": np.full(length, np.nan, dtype=float),
        "selector_regime_mean": np.full(length, np.nan, dtype=float),
        "selector_regime_ready": np.zeros(length, dtype=bool),
        "selector_cash_prediction": np.zeros(length, dtype=bool),
        "selector_skip_prediction": np.zeros(length, dtype=bool),
    }
    for regime in REGIME_NAMES:
        arrays[f"{regime}_n_raw"] = np.zeros(length, dtype=int)
        arrays[f"{regime}_n_eff"] = np.zeros(length, dtype=float)
        arrays[f"{regime}_weighted_label_sum"] = np.zeros(
            length, dtype=float
        )
        arrays[f"{regime}_weighted_squared_label_sum"] = np.zeros(
            length, dtype=float
        )
        arrays[f"{regime}_mean"] = np.full(length, np.nan, dtype=float)
        arrays[f"{regime}_ready"] = np.zeros(length, dtype=bool)
        arrays[f"{regime}_cash_selected"] = np.zeros(length, dtype=bool)
    return arrays


def _record_state(
    arrays: dict[str, np.ndarray],
    *,
    position: int,
    regime: str,
    state: EwRegimeState,
) -> None:
    arrays[f"{regime}_n_raw"][position] = state.n_raw
    arrays[f"{regime}_n_eff"][position] = state.n_eff
    arrays[f"{regime}_weighted_label_sum"][position] = (
        state.weighted_label_sum
    )
    arrays[f"{regime}_weighted_squared_label_sum"][position] = (
        state.weighted_squared_label_sum
    )
    arrays[f"{regime}_mean"][position] = state.mean
    arrays[f"{regime}_ready"][position] = state.ready
    arrays[f"{regime}_cash_selected"][position] = state.cash_selected


def replay_binary_regime_union_selector(
    frame: pd.DataFrame,
    *,
    learning_mode: str,
    frozen_cutoff: str | pd.Timestamp | None = None,
) -> BinaryRegimeSelectorReplay:
    """Replay the two EW states using only causally matured union lessons."""

    cutoff = _validate_learning_contract(learning_mode, frozen_cutoff)
    data = _canonical_market_frame(frame)
    signals = build_fixed_expert_signals(data)
    features = build_binary_regime_features(data)
    if not features.index.equals(data.index) or not signals.index.equals(data.index):
        raise ValueError("fixed signals and regime features must align exactly")

    length = len(data)
    arrays = _diagnostic_arrays(length)
    dates = data.index
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    raw_candidates = signals[
        "unfiltered_union_candidate_signal"
    ].to_numpy(dtype=bool)
    shadow_union = signals["unfiltered_union_signal"].to_numpy(dtype=bool)
    regime_values = features["risk_on"].to_numpy(dtype=bool)

    shadow_positions = np.flatnonzero(shadow_union)
    resolved_positions = shadow_positions[shadow_positions + 2 < length]
    arrays["shadow_matures_on_close"][resolved_positions] = date_values[
        resolved_positions + 2
    ]
    arrays["shadow_opportunity_risk_on"][shadow_positions] = regime_values[
        shadow_positions
    ]
    pending_positions = shadow_positions[shadow_positions + 2 >= length]
    arrays["shadow_pending"][pending_positions] = True

    states = initialize_regime_states()
    for current in range(length):
        signal_position = current - 2
        if signal_position >= 0 and shadow_union[signal_position]:
            label = (
                math.log(opens[signal_position + 1] / opens[current])
                + ROUND_TRIP_LOG_FRICTION
            )
            signal_risk_on = bool(regime_values[signal_position])
            regime = RISK_ON_REGIME if signal_risk_on else NOT_RISK_ON_REGIME
            arrays["shadow_matured_now"][current] = True
            arrays["shadow_signal_close"][current] = date_values[
                signal_position
            ]
            arrays["shadow_matured_signal_risk_on"][current] = signal_risk_on
            arrays["shadow_label_10bps"][current] = label
            allowed = bool(
                dates[signal_position] >= LESSON_MEMORY_START
                and (cutoff is None or dates[current] <= cutoff)
            )
            if allowed:
                default_cash = (
                    RISK_ON_STRUCTURAL_DEFAULT_CASH
                    if signal_risk_on
                    else NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH
                )
                states[regime] = update_ew_regime_state(
                    states[regime],
                    label,
                    structural_default_cash=default_cash,
                )
                arrays["shadow_lesson_added_now"][current] = True

        for regime in REGIME_NAMES:
            _record_state(
                arrays,
                position=current,
                regime=regime,
                state=states[regime],
            )

        if raw_candidates[current]:
            candidate_regime = (
                RISK_ON_REGIME
                if regime_values[current]
                else NOT_RISK_ON_REGIME
            )
            state = states[candidate_regime]
            arrays["selector_regime_n_eff"][current] = state.n_eff
            arrays["selector_regime_mean"][current] = state.mean
            arrays["selector_regime_ready"][current] = state.ready
            arrays["selector_cash_prediction"][current] = state.cash_selected
            arrays["selector_skip_prediction"][current] = not state.cash_selected

    selector_skip_signal = shadow_union & arrays["selector_skip_prediction"]
    learner_cash = shadow_union & arrays["selector_cash_prediction"]

    pending: list[PendingRegimeLesson] = []
    for position in pending_positions:
        remaining = int(position + 2 - (length - 1))
        entry = float(opens[position + 1]) if position + 1 < length else None
        pending.append(
            PendingRegimeLesson(
                signal_close=dates[position],
                sessions_until_maturity=remaining,
                risk_on=bool(regime_values[position]),
                entry_adjusted_open=entry,
                learning_eligible=bool(dates[position] >= LESSON_MEMORY_START),
            )
        )
    pending_tuple = tuple(pending)
    final_states = _strict_regime_state_mapping(states)

    result = signals.copy()
    result = result.join(features, how="left")
    result["union_candidate_signal"] = raw_candidates
    result["shadow_canonical_union_signal"] = shadow_union
    # This alias deliberately remains the continuous shadow union.  The
    # runner constructs the once-canonicalized 2005 account union itself.
    result["canonical_union_cash_signal"] = shadow_union
    for name, values in arrays.items():
        result[name] = values
    result["selector_skip_signal"] = selector_skip_signal
    result["learner_cash_signal"] = learner_cash
    result["target_exposure"] = np.where(learner_cash, 0.0, 1.0)
    result.attrs.update(
        {
            "learning_mode": learning_mode,
            "frozen_cutoff": (
                cutoff.date().isoformat() if cutoff is not None else None
            ),
            "post_cutoff_outcomes_used_for_learning": bool(
                learning_mode == CAUSAL_ONLINE_MODE
            ),
            "regime_feature_order": list(REGIME_FEATURE_COLUMNS),
            "risk_on_definition": (
                "spy adjusted-close 20-session return > 0 AND "
                "qqq adjusted-close 20-session return > 0"
            ),
            "lesson_maturity": "close t+2 before same-close prediction",
            "lesson_memory_start": LESSON_MEMORY_START.date().isoformat(),
            "round_trip_log_friction": ROUND_TRIP_LOG_FRICTION,
            "lesson_discount": LESSON_DISCOUNT,
            "minimum_effective_lessons": MIN_EFFECTIVE_LESSONS,
            "positive_mean_threshold": POSITIVE_MEAN_THRESHOLD,
            "negative_mean_threshold": NEGATIVE_MEAN_THRESHOLD,
            "risk_on_structural_default_cash": (
                RISK_ON_STRUCTURAL_DEFAULT_CASH
            ),
            "not_risk_on_structural_default_cash": (
                NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH
            ),
            "final_states": serialize_regime_states(final_states),
            "pending_lessons": serialize_pending_regime_lessons(
                pending_tuple
            ),
            "shadow_stream": (
                "continuous canonical fixed union; independent of selector action"
            ),
        }
    )
    return BinaryRegimeSelectorReplay(
        forecast=result,
        final_states=final_states,
        pending_lessons=pending_tuple,
    )


def build_binary_regime_union_selector_forecast(
    frame: pd.DataFrame,
    *,
    learning_mode: str,
    frozen_cutoff: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return the deterministic continuous forecast used by staged runners."""

    return replay_binary_regime_union_selector(
        frame,
        learning_mode=learning_mode,
        frozen_cutoff=frozen_cutoff,
    ).forecast


__all__ = [
    "BinaryRegimeSelectorReplay",
    "CAUSAL_ONLINE_MODE",
    "EwRegimeState",
    "FROZEN_CUTOFF_MODE",
    "LESSON_DISCOUNT",
    "MIN_EFFECTIVE_LESSONS",
    "NEGATIVE_MEAN_THRESHOLD",
    "NOT_RISK_ON_REGIME",
    "NOT_RISK_ON_STRUCTURAL_DEFAULT_CASH",
    "PENDING_SCHEMA_VERSION",
    "POSITIVE_MEAN_THRESHOLD",
    "PendingRegimeLesson",
    "REGIME_FEATURE_COLUMNS",
    "REGIME_LOOKBACK",
    "REGIME_NAMES",
    "RISK_ON_REGIME",
    "RISK_ON_STRUCTURAL_DEFAULT_CASH",
    "STATE_SCHEMA_VERSION",
    "build_binary_regime_features",
    "build_binary_regime_union_selector_forecast",
    "initialize_regime_state",
    "initialize_regime_states",
    "replay_binary_regime_union_selector",
    "resolve_regime_cash_selection",
    "restore_ew_regime_state",
    "restore_pending_regime_lessons",
    "restore_regime_states",
    "serialize_ew_regime_state",
    "serialize_pending_regime_lessons",
    "serialize_regime_states",
    "update_ew_regime_state",
]
