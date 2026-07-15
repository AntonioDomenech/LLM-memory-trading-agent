"""Causal discounted-Bayesian veto for the fixed exhaustion union.

This module contains no stage scoring or market-data acquisition.  It builds
the frozen close-time features, replays action-independent shadow lessons, and
emits an auditable continuous forecast for the sealed experiment runner.
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
    canonicalize_one_session_signals,
)


VETO_FEATURE_COLUMNS = (
    "feature_intercept",
    "feature_weak_only",
    "feature_expert_overlap",
    "feature_tail_strength",
    "feature_market_sentiment_10",
    "feature_market_sentiment_20",
    "feature_aapl_trend",
)
FEATURE_ORDER = VETO_FEATURE_COLUMNS
FEATURE_COUNT = len(VETO_FEATURE_COLUMNS)

COEFFICIENT_PRIOR_STANDARD_DEVIATION = 0.02
OBSERVATION_STANDARD_DEVIATION = 0.04
LESSON_DISCOUNT = 0.995
MIN_RAW_LESSONS = 40
MIN_EFFECTIVE_LESSONS = 30.0
CONFIDENCE_MULTIPLIER = 1.282
MINIMUM_PREDICTED_HARM = 0.001
INTRADAY_RANK_LOOKBACK = 126

STATE_SCHEMA_VERSION = "union-contextual-veto-state-v1"
PENDING_SCHEMA_VERSION = "union-contextual-veto-pending-v1"


@dataclass(frozen=True)
class BayesianVetoState:
    """Immutable sufficient state for the frozen discounted linear model."""

    n_raw: int
    n_eff: float
    wins: int
    A: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    label_sum: float
    squared_label_sum: float


@dataclass(frozen=True)
class BayesianVetoPrediction:
    beta: tuple[float, ...]
    mu: float
    se: float
    upper: float
    model_ready: bool
    veto: bool


@dataclass(frozen=True)
class PendingShadowLesson:
    """A canonical shadow signal whose t+2 adjusted open is not yet known."""

    signal_close: pd.Timestamp
    sessions_until_maturity: int
    feature_values: tuple[float, ...]
    entry_adjusted_open: float | None
    learning_eligible: bool


@dataclass(frozen=True)
class UnionContextualVetoReplay:
    forecast: pd.DataFrame
    final_state: BayesianVetoState
    pending_lessons: tuple[PendingShadowLesson, ...]


def _feature_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (FEATURE_COUNT,):
        raise ValueError(
            f"feature_values must have shape ({FEATURE_COUNT},), got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError("feature_values must be finite")
    return vector


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number != value or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return number


def _state_arrays(state: BayesianVetoState) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(state, BayesianVetoState):
        raise TypeError("state must be a BayesianVetoState")
    n_raw = _integer(state.n_raw, name="n_raw")
    wins = _integer(state.wins, name="wins")
    if wins > n_raw:
        raise ValueError("wins cannot exceed n_raw")
    scalars = np.asarray(
        [state.n_eff, state.label_sum, state.squared_label_sum], dtype=float
    )
    if not np.isfinite(scalars).all():
        raise ValueError("Bayesian state scalars must be finite")
    if state.n_eff < 0.0 or state.n_eff > n_raw + 1e-12:
        raise ValueError("n_eff must be between zero and n_raw")
    if state.squared_label_sum < 0.0:
        raise ValueError("squared_label_sum must be non-negative")
    A = np.asarray(state.A, dtype=float)
    b = np.asarray(state.b, dtype=float)
    if A.shape != (FEATURE_COUNT, FEATURE_COUNT):
        raise ValueError("A has the wrong shape")
    if b.shape != (FEATURE_COUNT,):
        raise ValueError("b has the wrong shape")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Bayesian state A and b must be finite")
    if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
        raise ValueError("Bayesian precision matrix A must be symmetric")
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Bayesian precision matrix A must be positive definite") from exc
    return A, b


def initialize_bayesian_veto_state() -> BayesianVetoState:
    prior = np.eye(FEATURE_COUNT, dtype=float) / (
        COEFFICIENT_PRIOR_STANDARD_DEVIATION**2
    )
    return BayesianVetoState(
        n_raw=0,
        n_eff=0.0,
        wins=0,
        A=tuple(tuple(float(value) for value in row) for row in prior),
        b=tuple(0.0 for _ in range(FEATURE_COUNT)),
        label_sum=0.0,
        squared_label_sum=0.0,
    )


def update_bayesian_veto_state(
    state: BayesianVetoState,
    feature_values: Sequence[float] | np.ndarray,
    label: float,
) -> BayesianVetoState:
    """Return a newly updated state; the supplied state is never mutated."""

    A, b = _state_arrays(state)
    x = _feature_vector(feature_values)
    y = float(label)
    if not math.isfinite(y):
        raise ValueError("label must be finite")
    prior = np.eye(FEATURE_COUNT, dtype=float) / (
        COEFFICIENT_PRIOR_STANDARD_DEVIATION**2
    )
    inverse_observation_variance = 1.0 / OBSERVATION_STANDARD_DEVIATION**2
    updated_A = (
        prior
        + LESSON_DISCOUNT * (A - prior)
        + np.outer(x, x) * inverse_observation_variance
    )
    updated_b = (
        LESSON_DISCOUNT * b + x * y * inverse_observation_variance
    )
    updated = BayesianVetoState(
        n_raw=state.n_raw + 1,
        n_eff=LESSON_DISCOUNT * state.n_eff + 1.0,
        wins=state.wins + int(y > 0.0),
        A=tuple(tuple(float(value) for value in row) for row in updated_A),
        b=tuple(float(value) for value in updated_b),
        label_sum=state.label_sum + y,
        squared_label_sum=state.squared_label_sum + y**2,
    )
    _state_arrays(updated)
    return updated


def should_veto(*, n_raw: int, n_eff: float, upper: float) -> bool:
    raw = _integer(n_raw, name="n_raw")
    effective = float(n_eff)
    bound = float(upper)
    if not math.isfinite(effective) or effective < 0.0:
        raise ValueError("n_eff must be finite and non-negative")
    if not math.isfinite(bound):
        raise ValueError("upper must be finite")
    return bool(
        raw >= MIN_RAW_LESSONS
        and effective >= MIN_EFFECTIVE_LESSONS
        and bound < -MINIMUM_PREDICTED_HARM
    )


def predict_bayesian_veto(
    state: BayesianVetoState,
    feature_values: Sequence[float] | np.ndarray,
) -> BayesianVetoPrediction:
    A, b = _state_arrays(state)
    x = _feature_vector(feature_values)
    try:
        beta = np.linalg.solve(A, b)
        covariance_times_x = np.linalg.solve(A, x)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Bayesian state could not be solved") from exc
    mu = float(x @ beta)
    variance = float(x @ covariance_times_x)
    if not math.isfinite(mu) or not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("Bayesian prediction is nonfinite or non-positive")
    se = math.sqrt(variance)
    upper = mu + CONFIDENCE_MULTIPLIER * se
    ready = bool(
        state.n_raw >= MIN_RAW_LESSONS
        and state.n_eff >= MIN_EFFECTIVE_LESSONS
    )
    veto = should_veto(n_raw=state.n_raw, n_eff=state.n_eff, upper=upper)
    return BayesianVetoPrediction(
        beta=tuple(float(value) for value in beta),
        mu=mu,
        se=se,
        upper=upper,
        model_ready=ready,
        veto=veto,
    )


def serialize_bayesian_veto_state(state: BayesianVetoState) -> dict[str, Any]:
    A, b = _state_arrays(state)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "feature_order": list(VETO_FEATURE_COLUMNS),
        "n_raw": int(state.n_raw),
        "n_eff": float(state.n_eff),
        "wins": int(state.wins),
        "A": [[float(value) for value in row] for row in A],
        "b": [float(value) for value in b],
        "label_sum": float(state.label_sum),
        "squared_label_sum": float(state.squared_label_sum),
    }


def restore_bayesian_veto_state(payload: Mapping[str, Any]) -> BayesianVetoState:
    if not isinstance(payload, Mapping):
        raise TypeError("state payload must be a mapping")
    required = {
        "schema_version",
        "feature_order",
        "n_raw",
        "n_eff",
        "wins",
        "A",
        "b",
        "label_sum",
        "squared_label_sum",
    }
    if set(payload) != required:
        raise ValueError("state payload has missing or unexpected fields")
    if payload["schema_version"] != STATE_SCHEMA_VERSION:
        raise ValueError("state payload schema_version is incompatible")
    if tuple(payload["feature_order"]) != VETO_FEATURE_COLUMNS:
        raise ValueError("state payload feature_order is incompatible")
    try:
        A = tuple(
            tuple(float(value) for value in row) for row in payload["A"]
        )
        b = tuple(float(value) for value in payload["b"])
        state = BayesianVetoState(
            n_raw=_integer(payload["n_raw"], name="n_raw"),
            n_eff=float(payload["n_eff"]),
            wins=_integer(payload["wins"], name="wins"),
            A=A,
            b=b,
            label_sum=float(payload["label_sum"]),
            squared_label_sum=float(payload["squared_label_sum"]),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("state payload contains invalid values") from exc
    _state_arrays(state)
    return state


def serialize_pending_shadow_lessons(
    pending_lessons: Sequence[PendingShadowLesson],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for lesson in pending_lessons:
        if not isinstance(lesson, PendingShadowLesson):
            raise TypeError("pending_lessons must contain PendingShadowLesson values")
        features = _feature_vector(lesson.feature_values)
        remaining = _integer(
            lesson.sessions_until_maturity,
            name="sessions_until_maturity",
            minimum=1,
        )
        if remaining > 2:
            raise ValueError("sessions_until_maturity cannot exceed two")
        signal_close = pd.Timestamp(lesson.signal_close)
        if pd.isna(signal_close):
            raise ValueError("pending signal_close must be valid")
        if signal_close.tzinfo is not None:
            signal_close = signal_close.tz_localize(None)
        entry = lesson.entry_adjusted_open
        if entry is not None and (not math.isfinite(float(entry)) or float(entry) <= 0):
            raise ValueError("pending entry_adjusted_open must be positive and finite")
        rows.append(
            {
                "signal_close": signal_close.isoformat(),
                "sessions_until_maturity": remaining,
                "feature_values": [float(value) for value in features],
                "entry_adjusted_open": None if entry is None else float(entry),
                "learning_eligible": bool(lesson.learning_eligible),
            }
        )
    return {
        "schema_version": PENDING_SCHEMA_VERSION,
        "feature_order": list(VETO_FEATURE_COLUMNS),
        "lessons": rows,
    }


def restore_pending_shadow_lessons(
    payload: Mapping[str, Any],
) -> tuple[PendingShadowLesson, ...]:
    if not isinstance(payload, Mapping):
        raise TypeError("pending payload must be a mapping")
    if set(payload) != {"schema_version", "feature_order", "lessons"}:
        raise ValueError("pending payload has missing or unexpected fields")
    if payload["schema_version"] != PENDING_SCHEMA_VERSION:
        raise ValueError("pending payload schema_version is incompatible")
    if tuple(payload["feature_order"]) != VETO_FEATURE_COLUMNS:
        raise ValueError("pending payload feature_order is incompatible")
    lessons = payload["lessons"]
    if not isinstance(lessons, list):
        raise ValueError("pending payload lessons must be a list")
    restored: list[PendingShadowLesson] = []
    keys = {
        "signal_close",
        "sessions_until_maturity",
        "feature_values",
        "entry_adjusted_open",
        "learning_eligible",
    }
    for row in lessons:
        if not isinstance(row, Mapping) or set(row) != keys:
            raise ValueError("pending lesson has missing or unexpected fields")
        entry = row["entry_adjusted_open"]
        lesson = PendingShadowLesson(
            signal_close=pd.Timestamp(row["signal_close"]),
            sessions_until_maturity=_integer(
                row["sessions_until_maturity"],
                name="sessions_until_maturity",
                minimum=1,
            ),
            feature_values=tuple(float(value) for value in row["feature_values"]),
            entry_adjusted_open=None if entry is None else float(entry),
            learning_eligible=bool(row["learning_eligible"]),
        )
        # Reuse the serializer's strict validation for each restored item.
        serialize_pending_shadow_lessons((lesson,))
        restored.append(lesson)
    return tuple(restored)


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
            raise ValueError("frozen_cutoff mode requires an inclusive frozen_cutoff")
        cutoff = pd.Timestamp(frozen_cutoff)
        if pd.isna(cutoff):
            raise ValueError("frozen_cutoff must be a valid timestamp")
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        return cutoff
    if frozen_cutoff is not None:
        raise ValueError("causal_online mode must not receive a frozen_cutoff")
    return None


def _aligned_signals(data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    value = signals.copy()
    value.index = pd.DatetimeIndex(pd.to_datetime(value.index, errors="raise")).tz_localize(None)
    value = value.sort_index()
    if not value.index.equals(data.index):
        raise ValueError("signals must align exactly with the market frame")
    required = {
        "contextual_virtual_signal",
        "weak_trend_virtual_signal",
        "unfiltered_union_candidate_signal",
        "unfiltered_union_signal",
        "contextual_spy_return_10",
        "contextual_qqq_return_10",
        "weak_trend_spy_return_20",
        "weak_trend_qqq_return_20",
        "weak_trend_aapl_sma_20",
    }
    missing = sorted(required.difference(value.columns))
    if missing:
        raise ValueError(f"signals are missing required columns: {missing}")
    return value


def build_union_contextual_veto_features(
    frame: pd.DataFrame,
    *,
    signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the exact seven frozen close-time features without normalization."""

    data = _canonical_market_frame(frame)
    fixed = build_fixed_expert_signals(data) if signals is None else signals
    fixed = _aligned_signals(data, fixed)
    intraday = data["aapl_close"] / data["aapl_open"] - 1.0
    intraday_values = intraday.to_numpy(dtype=float)
    ranks = np.full(len(data), np.nan, dtype=float)
    for position in range(INTRADAY_RANK_LOOKBACK, len(data)):
        prior = intraday_values[
            position - INTRADAY_RANK_LOOKBACK : position
        ]
        ranks[position] = float(
            np.count_nonzero(prior <= intraday_values[position])
        ) / float(INTRADAY_RANK_LOOKBACK)

    contextual = fixed["contextual_virtual_signal"].astype(bool)
    weak = fixed["weak_trend_virtual_signal"].astype(bool)
    market_10 = (
        pd.to_numeric(fixed["contextual_spy_return_10"], errors="coerce")
        + pd.to_numeric(fixed["contextual_qqq_return_10"], errors="coerce")
    ) / 2.0
    market_20 = (
        pd.to_numeric(fixed["weak_trend_spy_return_20"], errors="coerce")
        + pd.to_numeric(fixed["weak_trend_qqq_return_20"], errors="coerce")
    ) / 2.0
    sma_20 = pd.to_numeric(fixed["weak_trend_aapl_sma_20"], errors="coerce")
    result = pd.DataFrame(
        {
            "aapl_intraday_rank_126": ranks,
            "feature_intercept": 1.0,
            "feature_weak_only": (weak & ~contextual).astype(float),
            "feature_expert_overlap": (weak & contextual).astype(float),
            "feature_tail_strength": np.clip((ranks - 0.90) / 0.10, 0.0, 1.0),
            "feature_market_sentiment_10": np.clip(market_10 / 0.10, -1.0, 1.0),
            "feature_market_sentiment_20": np.clip(market_20 / 0.15, -1.0, 1.0),
            "feature_aapl_trend": np.clip(
                (data["aapl_adj_close"] / sma_20 - 1.0) / 0.10,
                -1.0,
                1.0,
            ),
        },
        index=data.index,
    )
    candidates = fixed["unfiltered_union_candidate_signal"].astype(bool)
    candidate_values = result.loc[candidates, list(VETO_FEATURE_COLUMNS)].to_numpy(
        dtype=float
    )
    if not np.isfinite(candidate_values).all():
        raise ValueError("a raw union candidate has a missing or nonfinite feature")
    result.attrs["feature_order"] = list(VETO_FEATURE_COLUMNS)
    result.attrs["rank_reference"] = "prior 126 completed sessions; current excluded"
    return result


def _diagnostic_arrays(length: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "shadow_matures_on_close": np.full(length, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "shadow_matured_now": np.zeros(length, dtype=bool),
        "shadow_signal_close": np.full(length, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "shadow_label_10bps": np.full(length, np.nan, dtype=float),
        "shadow_lesson_added_now": np.zeros(length, dtype=bool),
        "shadow_pending": np.zeros(length, dtype=bool),
        "n_raw": np.zeros(length, dtype=int),
        "n_eff": np.zeros(length, dtype=float),
        "wins": np.zeros(length, dtype=int),
        "label_sum": np.zeros(length, dtype=float),
        "squared_label_sum": np.zeros(length, dtype=float),
        "model_mu": np.full(length, np.nan, dtype=float),
        "model_se": np.full(length, np.nan, dtype=float),
        "model_upper": np.full(length, np.nan, dtype=float),
        "model_ready": np.zeros(length, dtype=bool),
        "model_veto_prediction": np.zeros(length, dtype=bool),
        "veto": np.zeros(length, dtype=bool),
    }
    for row in range(FEATURE_COUNT):
        for column in range(row, FEATURE_COUNT):
            arrays[f"state_A_{row}_{column}"] = np.zeros(length, dtype=float)
        arrays[f"state_b_{row}"] = np.zeros(length, dtype=float)
        arrays[f"model_beta_{row}"] = np.full(length, np.nan, dtype=float)
    return arrays


def replay_union_contextual_veto(
    frame: pd.DataFrame,
    *,
    learning_mode: str,
    frozen_cutoff: str | pd.Timestamp | None = None,
) -> UnionContextualVetoReplay:
    """Replay matured shadow lessons and return forecast plus checkpoint state."""

    cutoff = _validate_learning_contract(learning_mode, frozen_cutoff)
    data = _canonical_market_frame(frame)
    signals = build_fixed_expert_signals(data)
    features = build_union_contextual_veto_features(data, signals=signals)
    length = len(data)
    arrays = _diagnostic_arrays(length)
    dates = data.index
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    raw_candidates = signals["unfiltered_union_candidate_signal"].to_numpy(dtype=bool)
    shadow_union = signals["unfiltered_union_signal"].to_numpy(dtype=bool)
    feature_matrix = features.loc[:, list(VETO_FEATURE_COLUMNS)].to_numpy(dtype=float)

    shadow_positions = np.flatnonzero(shadow_union)
    resolved_positions = shadow_positions[shadow_positions + 2 < length]
    arrays["shadow_matures_on_close"][resolved_positions] = date_values[
        resolved_positions + 2
    ]
    pending_positions = shadow_positions[shadow_positions + 2 >= length]
    arrays["shadow_pending"][pending_positions] = True

    state = initialize_bayesian_veto_state()
    for current in range(length):
        signal_position = current - 2
        if signal_position >= 0 and shadow_union[signal_position]:
            label = (
                math.log(opens[signal_position + 1] / opens[current])
                + ROUND_TRIP_LOG_FRICTION
            )
            arrays["shadow_matured_now"][current] = True
            arrays["shadow_signal_close"][current] = date_values[signal_position]
            arrays["shadow_label_10bps"][current] = label
            allowed = bool(
                dates[signal_position] >= LESSON_MEMORY_START
                and (cutoff is None or dates[current] <= cutoff)
            )
            if allowed:
                state = update_bayesian_veto_state(
                    state, feature_matrix[signal_position], label
                )
                arrays["shadow_lesson_added_now"][current] = True

        A, b = _state_arrays(state)
        arrays["n_raw"][current] = state.n_raw
        arrays["n_eff"][current] = state.n_eff
        arrays["wins"][current] = state.wins
        arrays["label_sum"][current] = state.label_sum
        arrays["squared_label_sum"][current] = state.squared_label_sum
        arrays["model_ready"][current] = bool(
            state.n_raw >= MIN_RAW_LESSONS
            and state.n_eff >= MIN_EFFECTIVE_LESSONS
        )
        for row in range(FEATURE_COUNT):
            for column in range(row, FEATURE_COUNT):
                arrays[f"state_A_{row}_{column}"][current] = A[row, column]
            arrays[f"state_b_{row}"][current] = b[row]

        if raw_candidates[current]:
            prediction = predict_bayesian_veto(state, feature_matrix[current])
            arrays["model_mu"][current] = prediction.mu
            arrays["model_se"][current] = prediction.se
            arrays["model_upper"][current] = prediction.upper
            arrays["model_veto_prediction"][current] = prediction.veto
            for feature_index, beta in enumerate(prediction.beta):
                arrays[f"model_beta_{feature_index}"][current] = beta

    # Veto is an action diagnostic.  Blocked raw t+1 candidates are still
    # scored above, but only the already-canonical union can be vetoed.
    arrays["veto"] = shadow_union & arrays["model_veto_prediction"]
    learner_cash = shadow_union & ~arrays["veto"]

    pending: list[PendingShadowLesson] = []
    for position in pending_positions:
        entry = float(opens[position + 1]) if position + 1 < length else None
        pending.append(
            PendingShadowLesson(
                signal_close=dates[position],
                sessions_until_maturity=int(position + 2 - (length - 1)),
                feature_values=tuple(float(value) for value in feature_matrix[position]),
                entry_adjusted_open=entry,
                learning_eligible=bool(dates[position] >= LESSON_MEMORY_START),
            )
        )

    result = signals.copy()
    result = result.join(features, how="left")
    result["union_candidate_signal"] = raw_candidates
    result["shadow_canonical_union_signal"] = shadow_union
    result["canonical_union_cash_signal"] = shadow_union
    for name, values in arrays.items():
        result[name] = values
    result["learner_cash_signal"] = learner_cash
    result["target_exposure"] = np.where(learner_cash, 0.0, 1.0)
    result.attrs.update(
        {
            "learning_mode": learning_mode,
            "frozen_cutoff": cutoff.date().isoformat() if cutoff is not None else None,
            "post_cutoff_outcomes_used_for_learning": bool(
                learning_mode == CAUSAL_ONLINE_MODE
            ),
            "feature_order": list(VETO_FEATURE_COLUMNS),
            "lesson_maturity": "close t+2 before same-close prediction",
            "lesson_memory_start": LESSON_MEMORY_START.date().isoformat(),
            "round_trip_log_friction": ROUND_TRIP_LOG_FRICTION,
            "coefficient_prior_standard_deviation": COEFFICIENT_PRIOR_STANDARD_DEVIATION,
            "observation_standard_deviation": OBSERVATION_STANDARD_DEVIATION,
            "lesson_discount": LESSON_DISCOUNT,
            "minimum_raw_lessons": MIN_RAW_LESSONS,
            "minimum_effective_lessons": MIN_EFFECTIVE_LESSONS,
            "confidence_multiplier": CONFIDENCE_MULTIPLIER,
            "minimum_predicted_harm": MINIMUM_PREDICTED_HARM,
            "final_state": serialize_bayesian_veto_state(state),
            "pending_lessons": serialize_pending_shadow_lessons(pending),
            "shadow_stream": "continuous canonical union; independent of learner action",
        }
    )
    return UnionContextualVetoReplay(
        forecast=result,
        final_state=state,
        pending_lessons=tuple(pending),
    )


def build_union_contextual_veto_forecast(
    frame: pd.DataFrame,
    *,
    learning_mode: str,
    frozen_cutoff: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return the deterministic continuous forecast used by staged runners."""

    return replay_union_contextual_veto(
        frame,
        learning_mode=learning_mode,
        frozen_cutoff=frozen_cutoff,
    ).forecast
