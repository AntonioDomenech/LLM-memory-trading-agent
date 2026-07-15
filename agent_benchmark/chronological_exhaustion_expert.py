from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FROZEN_CUTOFF_MODE = "frozen_cutoff"
CAUSAL_ONLINE_MODE = "causal_online"
LEARNING_MODES = frozenset({FROZEN_CUTOFF_MODE, CAUSAL_ONLINE_MODE})

EXPERT_NAMES = ("contextual", "weak_trend")

CONTEXTUAL_PERCENTILE = 0.90
WEAK_TREND_PERCENTILE = 0.925
INTRADAY_LOOKBACK = 126
CONTEXTUAL_MARKET_LOOKBACK = 10
WEAK_TREND_MARKET_LOOKBACK = 20
WEAK_TREND_SMA_LOOKBACK = 20

MIN_MATURED_EPISODES = 12
MIN_WIN_PROBABILITY = 0.55
MIN_PROBABILITY_LOWER_BOUND = 0.50
PROBABILITY_AND_EDGE_Z = 0.842
EDGE_PRIOR_STRENGTH = 8.0
EDGE_PRIOR_SCALE = 0.04
PER_SIDE_FRICTION = 0.001
LESSON_MEMORY_START = pd.Timestamp("2000-01-01")
ROUND_TRIP_LOG_FRICTION = math.log(
    (1.0 - PER_SIDE_FRICTION) / (1.0 + PER_SIDE_FRICTION)
)

REQUIRED_COLUMNS = (
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "spy_adj_close",
    "qqq_adj_close",
)


@dataclass(frozen=True)
class ExpertPosterior:
    matured_count: int
    matured_wins: int
    sum_net_edge: float
    sum_squared_net_edge: float
    alpha: float
    beta: float
    cash_win_probability: float
    cash_win_probability_lower: float
    edge_mean: float
    edge_second_moment: float
    edge_standard_error: float
    edge_lower_score: float
    count_gate: bool
    probability_gate: bool
    probability_lower_gate: bool
    edge_lower_score_gate: bool
    trusted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "matured_count": self.matured_count,
            "matured_wins": self.matured_wins,
            "sum_net_edge": self.sum_net_edge,
            "sum_squared_net_edge": self.sum_squared_net_edge,
            "alpha": self.alpha,
            "beta": self.beta,
            "cash_win_probability": self.cash_win_probability,
            "cash_win_probability_lower": self.cash_win_probability_lower,
            "edge_mean": self.edge_mean,
            "edge_second_moment": self.edge_second_moment,
            "edge_standard_error": self.edge_standard_error,
            "edge_lower_score": self.edge_lower_score,
            "count_gate": self.count_gate,
            "probability_gate": self.probability_gate,
            "probability_lower_gate": self.probability_lower_gate,
            "edge_lower_score_gate": self.edge_lower_score_gate,
            "trusted": self.trusted,
        }


def posterior_from_sufficient_statistics(
    *,
    count: int,
    wins: int,
    sum_net_edge: float,
    sum_squared_net_edge: float,
) -> ExpertPosterior:
    """Apply the experiment's frozen posterior and trust gates exactly."""

    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError("count must be a non-negative integer")
    if isinstance(wins, bool) or int(wins) != wins or wins < 0 or wins > count:
        raise ValueError("wins must be an integer between zero and count")
    if not math.isfinite(float(sum_net_edge)):
        raise ValueError("sum_net_edge must be finite")
    if (
        not math.isfinite(float(sum_squared_net_edge))
        or float(sum_squared_net_edge) < 0.0
    ):
        raise ValueError("sum_squared_net_edge must be finite and non-negative")

    count = int(count)
    wins = int(wins)
    sum_net_edge = float(sum_net_edge)
    sum_squared_net_edge = float(sum_squared_net_edge)
    alpha = float(wins + 1)
    beta = float(count - wins + 1)
    probability = alpha / (alpha + beta)
    probability_variance = alpha * beta / (
        (alpha + beta) ** 2 * (alpha + beta + 1.0)
    )
    probability_lower = probability - PROBABILITY_AND_EDGE_Z * math.sqrt(
        probability_variance
    )
    edge_denominator = count + EDGE_PRIOR_STRENGTH
    edge_mean = sum_net_edge / edge_denominator
    edge_second = (
        sum_squared_net_edge
        + EDGE_PRIOR_STRENGTH * EDGE_PRIOR_SCALE**2
    ) / edge_denominator
    edge_standard_error = math.sqrt(edge_second / edge_denominator)
    edge_lower_score = edge_mean - PROBABILITY_AND_EDGE_Z * edge_standard_error

    count_gate = count >= MIN_MATURED_EPISODES
    probability_gate = probability >= MIN_WIN_PROBABILITY
    probability_lower_gate = probability_lower > MIN_PROBABILITY_LOWER_BOUND
    edge_lower_score_gate = edge_lower_score > 0.0
    trusted = bool(
        count_gate
        and probability_gate
        and probability_lower_gate
        and edge_lower_score_gate
    )
    return ExpertPosterior(
        matured_count=count,
        matured_wins=wins,
        sum_net_edge=sum_net_edge,
        sum_squared_net_edge=sum_squared_net_edge,
        alpha=alpha,
        beta=beta,
        cash_win_probability=probability,
        cash_win_probability_lower=probability_lower,
        edge_mean=edge_mean,
        edge_second_moment=edge_second,
        edge_standard_error=edge_standard_error,
        edge_lower_score=edge_lower_score,
        count_gate=count_gate,
        probability_gate=probability_gate,
        probability_lower_gate=probability_lower_gate,
        edge_lower_score_gate=edge_lower_score_gate,
        trusted=trusted,
    )


def canonicalize_one_session_signals(signal: pd.Series) -> pd.Series:
    """Accept a signal unless the same policy accepted the preceding session."""

    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series")
    if signal.index.has_duplicates:
        raise ValueError("signal index must not contain duplicates")
    if signal.isna().any():
        raise ValueError("signal must not contain missing values")
    if not signal.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("signal must contain booleans only")

    raw = signal.to_numpy(dtype=bool)
    accepted = np.zeros(len(raw), dtype=bool)
    previous_accepted = -2
    for position, active in enumerate(raw):
        if active and position != previous_accepted + 1:
            accepted[position] = True
            previous_accepted = position
    return pd.Series(accepted, index=signal.index, name=signal.name, dtype=bool)


def _canonical_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    if "date" in clean.columns:
        clean["date"] = pd.to_datetime(clean["date"], errors="raise")
        clean = clean.set_index("date")
    clean.index = pd.DatetimeIndex(
        pd.to_datetime(clean.index, errors="raise")
    ).tz_localize(None)
    clean = clean.sort_index()
    if clean.empty:
        raise ValueError("market frame must not be empty")
    if clean.index.has_duplicates:
        raise ValueError("market frame must not contain duplicate sessions")
    missing = sorted(set(REQUIRED_COLUMNS).difference(clean.columns))
    if missing:
        raise ValueError(f"market frame is missing required columns: {missing}")
    for column in REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="raise").astype(float)
    values = clean.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("market prices must be finite")
    if (values <= 0.0).any():
        raise ValueError("market prices must be strictly positive")
    clean["aapl_adj_open"] = (
        clean["aapl_open"] * clean["aapl_adj_close"] / clean["aapl_close"]
    )
    if (
        ~np.isfinite(clean["aapl_adj_open"].to_numpy(dtype=float))
        | (clean["aapl_adj_open"].to_numpy(dtype=float) <= 0.0)
    ).any():
        raise ValueError("adjusted AAPL open could not be constructed")
    return clean.loc[:, [*REQUIRED_COLUMNS, "aapl_adj_open"]]


def build_fixed_expert_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the two fixed expert signals and identical unfiltered ablations.

    Every column is knowable at the completed close on its row. Targets are
    close-time targets and therefore take effect at the following adjusted
    open in the shared execution ledger.
    """

    data = _canonical_market_frame(frame)
    intraday = data["aapl_close"] / data["aapl_open"] - 1.0
    contextual_percentile = intraday.rolling(
        INTRADAY_LOOKBACK,
        min_periods=INTRADAY_LOOKBACK,
    ).quantile(CONTEXTUAL_PERCENTILE).shift(1)
    weak_trend_percentile = intraday.rolling(
        INTRADAY_LOOKBACK,
        min_periods=INTRADAY_LOOKBACK,
    ).quantile(WEAK_TREND_PERCENTILE).shift(1)
    contextual_spy_return = data["spy_adj_close"].pct_change(
        CONTEXTUAL_MARKET_LOOKBACK,
        fill_method=None,
    )
    contextual_qqq_return = data["qqq_adj_close"].pct_change(
        CONTEXTUAL_MARKET_LOOKBACK,
        fill_method=None,
    )
    weak_trend_spy_return = data["spy_adj_close"].pct_change(
        WEAK_TREND_MARKET_LOOKBACK,
        fill_method=None,
    )
    weak_trend_qqq_return = data["qqq_adj_close"].pct_change(
        WEAK_TREND_MARKET_LOOKBACK,
        fill_method=None,
    )
    weak_trend_aapl_sma = data["aapl_adj_close"].rolling(
        WEAK_TREND_SMA_LOOKBACK,
        min_periods=WEAK_TREND_SMA_LOOKBACK,
    ).mean()

    contextual_ready = (
        contextual_percentile.notna()
        & contextual_spy_return.notna()
        & contextual_qqq_return.notna()
    )
    weak_trend_ready = (
        weak_trend_percentile.notna()
        & weak_trend_spy_return.notna()
        & weak_trend_qqq_return.notna()
        & weak_trend_aapl_sma.notna()
    )
    contextual_raw = (
        contextual_ready
        & (intraday > contextual_percentile)
        & (contextual_spy_return < 0.0)
        & (contextual_qqq_return < 0.0)
    ).astype(bool)
    weak_trend_raw = (
        weak_trend_ready
        & (intraday > weak_trend_percentile)
        & (weak_trend_spy_return < 0.0)
        & (weak_trend_qqq_return < 0.0)
        & (data["aapl_adj_close"] < weak_trend_aapl_sma)
    ).astype(bool)
    contextual_virtual = canonicalize_one_session_signals(contextual_raw)
    weak_trend_virtual = canonicalize_one_session_signals(weak_trend_raw)
    union_candidate = (contextual_virtual | weak_trend_virtual).astype(bool)
    union_signal = canonicalize_one_session_signals(union_candidate)
    stage_outcome_available = pd.Series(False, index=data.index, dtype=bool)
    if len(stage_outcome_available) > 2:
        stage_outcome_available.iloc[:-2] = True
    contextual_actionable = contextual_virtual & stage_outcome_available
    weak_trend_actionable = weak_trend_virtual & stage_outcome_available
    union_actionable = union_signal & stage_outcome_available

    result = pd.DataFrame(
        {
            "aapl_intraday_return": intraday,
            "contextual_prior_intraday_percentile": contextual_percentile,
            "contextual_spy_return_10": contextual_spy_return,
            "contextual_qqq_return_10": contextual_qqq_return,
            "contextual_ready": contextual_ready.astype(bool),
            "contextual_raw_signal": contextual_raw,
            "contextual_virtual_signal": contextual_virtual,
            "contextual_virtual_signal_blocked": (
                contextual_raw & ~contextual_virtual
            ),
            "contextual_virtual_signal_pending_stage_outcome": (
                contextual_virtual & ~stage_outcome_available
            ),
            "weak_trend_prior_intraday_percentile": weak_trend_percentile,
            "weak_trend_spy_return_20": weak_trend_spy_return,
            "weak_trend_qqq_return_20": weak_trend_qqq_return,
            "weak_trend_aapl_sma_20": weak_trend_aapl_sma,
            "weak_trend_ready": weak_trend_ready.astype(bool),
            "weak_trend_raw_signal": weak_trend_raw,
            "weak_trend_virtual_signal": weak_trend_virtual,
            "weak_trend_virtual_signal_blocked": (
                weak_trend_raw & ~weak_trend_virtual
            ),
            "weak_trend_virtual_signal_pending_stage_outcome": (
                weak_trend_virtual & ~stage_outcome_available
            ),
            "unfiltered_union_candidate_signal": union_candidate,
            "unfiltered_union_signal": union_signal,
            "unfiltered_union_signal_blocked": union_candidate & ~union_signal,
            "unfiltered_union_signal_pending_stage_outcome": (
                union_signal & ~stage_outcome_available
            ),
            "stage_outcome_available": stage_outcome_available,
            "always_long_target_exposure": 1.0,
            "unfiltered_contextual_target_exposure": np.where(
                contextual_actionable, 0.0, 1.0
            ),
            "unfiltered_weak_trend_target_exposure": np.where(
                weak_trend_actionable, 0.0, 1.0
            ),
            "unfiltered_union_target_exposure": np.where(
                union_actionable, 0.0, 1.0
            ),
        },
        index=data.index,
    )
    result.attrs.update(
        {
            "signal_time": "completed close t",
            "fill_time": "adjusted AAPL open t+1",
            "episode_exit_time": "adjusted AAPL open t+2",
            "fixed_experts": list(EXPERT_NAMES),
        }
    )
    return result


def build_unfiltered_expert_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Return close-time targets for the four required policy ablations."""

    signals = build_fixed_expert_signals(frame)
    columns = [
        "always_long_target_exposure",
        "unfiltered_contextual_target_exposure",
        "unfiltered_weak_trend_target_exposure",
        "unfiltered_union_target_exposure",
    ]
    result = signals.loc[:, columns].copy()
    result.attrs.update(signals.attrs)
    return result


def _empty_diagnostic_arrays(length: int) -> dict[str, np.ndarray]:
    return {
        "episode_matures_on_close": np.full(length, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "matured_episode_now": np.zeros(length, dtype=bool),
        "matured_signal_close": np.full(length, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "matured_raw_edge": np.full(length, np.nan, dtype=float),
        "matured_net_edge": np.full(length, np.nan, dtype=float),
        "lesson_added_now": np.zeros(length, dtype=bool),
        "matured_count": np.zeros(length, dtype=int),
        "matured_wins": np.zeros(length, dtype=int),
        "sum_net_edge": np.zeros(length, dtype=float),
        "sum_squared_net_edge": np.zeros(length, dtype=float),
        "posterior_alpha": np.zeros(length, dtype=float),
        "posterior_beta": np.zeros(length, dtype=float),
        "posterior_cash_win_probability": np.zeros(length, dtype=float),
        "posterior_cash_win_probability_lower": np.zeros(length, dtype=float),
        "posterior_edge_mean": np.zeros(length, dtype=float),
        "posterior_edge_second_moment": np.zeros(length, dtype=float),
        "posterior_edge_standard_error": np.zeros(length, dtype=float),
        "posterior_edge_lower_score": np.zeros(length, dtype=float),
        "count_gate": np.zeros(length, dtype=bool),
        "probability_gate": np.zeros(length, dtype=bool),
        "probability_lower_gate": np.zeros(length, dtype=bool),
        "edge_lower_score_gate": np.zeros(length, dtype=bool),
        "posterior_trusted": np.zeros(length, dtype=bool),
        "trusted_signal": np.zeros(length, dtype=bool),
    }


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


def build_chronological_exhaustion_forecast(
    frame: pd.DataFrame,
    *,
    learning_mode: str,
    frozen_cutoff: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Replay the fixed experts with strictly matured outcome memory.

    A canonical virtual signal at close ``t`` receives no outcome until close
    ``t+2``. At that close, and before that close's prediction, its adjusted
    open-to-open cash advantage is admitted to memory if the selected learning
    mode permits it. The learned account accepts a trusted expert signal for
    one session and always ignores the immediately following session's action
    candidates, without suppressing either expert's virtual lesson stream.
    """

    cutoff = _validate_learning_contract(learning_mode, frozen_cutoff)
    data = _canonical_market_frame(frame)
    signals = build_fixed_expert_signals(data)
    length = len(data)
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    index_values = data.index.to_numpy(dtype="datetime64[ns]")
    diagnostics = {
        expert: _empty_diagnostic_arrays(length) for expert in EXPERT_NAMES
    }

    for expert in EXPERT_NAMES:
        arrays = diagnostics[expert]
        virtual = signals[f"{expert}_virtual_signal"].to_numpy(dtype=bool)
        accepted_positions = np.flatnonzero(virtual)
        positions_with_outcomes = accepted_positions[accepted_positions + 2 < length]
        arrays["episode_matures_on_close"][positions_with_outcomes] = index_values[
            positions_with_outcomes + 2
        ]

        count = 0
        wins = 0
        edge_sum = 0.0
        squared_edge_sum = 0.0
        for current in range(length):
            signal_position = current - 2
            if signal_position >= 0 and virtual[signal_position]:
                raw_edge = math.log(opens[signal_position + 1] / opens[current])
                net_edge = raw_edge + ROUND_TRIP_LOG_FRICTION
                arrays["matured_episode_now"][current] = True
                arrays["matured_signal_close"][current] = index_values[
                    signal_position
                ]
                arrays["matured_raw_edge"][current] = raw_edge
                arrays["matured_net_edge"][current] = net_edge
                lesson_is_allowed = (
                    data.index[signal_position] >= LESSON_MEMORY_START
                    and (cutoff is None or data.index[current] <= cutoff)
                )
                if lesson_is_allowed:
                    count += 1
                    wins += int(net_edge > 0.0)
                    edge_sum += net_edge
                    squared_edge_sum += net_edge**2
                    arrays["lesson_added_now"][current] = True

            posterior = posterior_from_sufficient_statistics(
                count=count,
                wins=wins,
                sum_net_edge=edge_sum,
                sum_squared_net_edge=squared_edge_sum,
            )
            arrays["matured_count"][current] = posterior.matured_count
            arrays["matured_wins"][current] = posterior.matured_wins
            arrays["sum_net_edge"][current] = posterior.sum_net_edge
            arrays["sum_squared_net_edge"][current] = (
                posterior.sum_squared_net_edge
            )
            arrays["posterior_alpha"][current] = posterior.alpha
            arrays["posterior_beta"][current] = posterior.beta
            arrays["posterior_cash_win_probability"][current] = (
                posterior.cash_win_probability
            )
            arrays["posterior_cash_win_probability_lower"][current] = (
                posterior.cash_win_probability_lower
            )
            arrays["posterior_edge_mean"][current] = posterior.edge_mean
            arrays["posterior_edge_second_moment"][current] = (
                posterior.edge_second_moment
            )
            arrays["posterior_edge_standard_error"][current] = (
                posterior.edge_standard_error
            )
            arrays["posterior_edge_lower_score"][current] = (
                posterior.edge_lower_score
            )
            arrays["count_gate"][current] = posterior.count_gate
            arrays["probability_gate"][current] = posterior.probability_gate
            arrays["probability_lower_gate"][current] = (
                posterior.probability_lower_gate
            )
            arrays["edge_lower_score_gate"][current] = (
                posterior.edge_lower_score_gate
            )
            arrays["posterior_trusted"][current] = posterior.trusted
            arrays["trusted_signal"][current] = bool(
                virtual[current] and posterior.trusted
            )

    result = signals.copy()
    for expert, arrays in diagnostics.items():
        for diagnostic_name, values in arrays.items():
            result[f"{expert}_{diagnostic_name}"] = values

    trusted_candidate = np.zeros(length, dtype=bool)
    for expert in EXPERT_NAMES:
        trusted_candidate |= diagnostics[expert]["trusted_signal"]
    stage_outcome_available = signals["stage_outcome_available"].to_numpy(dtype=bool)
    actionable_trusted_candidate = trusted_candidate & stage_outcome_available
    combined_cash_signal = canonicalize_one_session_signals(
        pd.Series(
            actionable_trusted_candidate,
            index=data.index,
            name="combined_cash_signal",
            dtype=bool,
        )
    )
    combined_cash_values = combined_cash_signal.to_numpy(dtype=bool)
    result["combined_trusted_candidate_signal"] = trusted_candidate
    result["combined_trusted_candidate_pending_stage_outcome"] = (
        trusted_candidate & ~stage_outcome_available
    )
    result["combined_cash_signal"] = combined_cash_values
    result["combined_signal_ignored_due_to_nonstacking"] = (
        actionable_trusted_candidate & ~combined_cash_values
    )
    result["target_exposure"] = np.where(combined_cash_values, 0.0, 1.0)
    result.attrs.update(
        {
            "learning_mode": learning_mode,
            "frozen_cutoff": cutoff.date().isoformat() if cutoff is not None else None,
            "post_cutoff_outcomes_used_for_learning": bool(
                learning_mode == CAUSAL_ONLINE_MODE
            ),
            "lesson_maturity": "close t+2 after adjusted open t+2",
            "round_trip_log_friction": ROUND_TRIP_LOG_FRICTION,
            "per_side_friction": PER_SIDE_FRICTION,
            "minimum_matured_episodes": MIN_MATURED_EPISODES,
            "lesson_memory_start": LESSON_MEMORY_START.date().isoformat(),
            "fixed_experts": list(EXPERT_NAMES),
        }
    )
    return result
