"""Pure scoring and gate logic for contextual expert aggregation.

This module is deliberately acquisition-free.  It accepts already constructed
ledger, episode, and XOR frames and turns them into deterministic JSON-friendly
metrics.  It does not import the legacy simulators because those modules import
market-data acquisition libraries at module load time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from . import contextual_expert_aggregation_ledger as _ledger


BASE_COST_NAME = "base_5bps"
STRESS_COST_NAME = "stress_10bps"
COST_NAMES = (BASE_COST_NAME, STRESS_COST_NAME)
COST_BPS: Mapping[str, float] = {
    BASE_COST_NAME: 5.0,
    STRESS_COST_NAME: 10.0,
}
INITIAL_CASH = _ledger.INITIAL_CASH

DEVELOPMENT_YEARS = tuple(range(2005, 2019))
DEVELOPMENT_FOLDS: Mapping[str, tuple[int, ...]] = {
    "2005_2006": (2005, 2006),
    "2007_2008": (2007, 2008),
    "2009_2010": (2009, 2010),
    "2011_2012": (2011, 2012),
    "2013_2014": (2013, 2014),
    "2015_2016": (2015, 2016),
    "2017_2018": (2017, 2018),
}
CONFIRMATION_YEARS = tuple(range(2019, 2024))
CONFIRMATION_ACCOUNT_YEARS = tuple(range(2005, 2024))

FIXED_COMPARATOR_NAMES = (
    "always_long",
    "exact_union_cash",
    "contextual_only",
    "weak_trend_only",
)
ABLATION_COMPARISON_NAMES = (
    "online_minus_frozen_2018",
    "full_minus_global_only",
    "full_minus_lifetime_only",
)
XOR_ORIENTATIONS = frozenset(
    {
        _ledger.PRIMARY_CASH_COMPARATOR_LONG,
        _ledger.PRIMARY_LONG_COMPARATOR_CASH,
    }
)

MIN_ACTIVE_LOG_EDGE = 0.001
STRICT_INCREMENTAL_EDGE = 0.0001
RECONCILIATION_TOLERANCE = _ledger.RECONCILIATION_TOLERANCE

LEDGER_COLUMNS = _ledger.LEDGER_COLUMNS
EPISODE_COLUMNS = _ledger.EPISODE_COLUMNS
XOR_COLUMNS = _ledger.XOR_COLUMNS
POLICY_EVIDENCE_KEYS = frozenset(
    {"strategy_ledger", "benchmark_ledger", "complete_episodes"}
)
PAIRWISE_EVIDENCE_KEYS = frozenset(
    set(POLICY_EVIDENCE_KEYS) | {"learner_minus_comparator_xor"}
)

COMMON_INTEGRITY_CHECKS = frozenset(
    {
        "authorized_price_and_parent_provenance_exact",
        "fixed_expert_and_union_prefix_exact",
        "one_lesson_per_opportunity_exact",
        "causal_maturity_and_update_order_exact",
        "state_probability_action_replay_exact",
        "checkpoint_pending_and_cooldown_exact",
        "binary_union_subset_actions_exact",
        "single_continuous_account_exact",
        "all_ledgers_unleveraged_exact",
        "same_actions_at_both_costs_exact",
        "always_long_equals_benchmark_exact",
        "ledger_episode_and_xor_reconciliation_exact",
        "zero_network_news_llm_api_and_external_cost",
    }
)
DEVELOPMENT_INTEGRITY_CHECKS = COMMON_INTEGRITY_CHECKS
CONFIRMATION_INTEGRITY_CHECKS = frozenset(
    set(COMMON_INTEGRITY_CHECKS)
    | {
        "development_checkpoint_continuity_exact",
        "online_frozen_forecast_prefix_exact",
        "frozen_post_cutoff_state_unchanged",
        "confirmation_attempt_authorization_exact",
    }
)


class ContextualExpertAggregationEvaluationError(RuntimeError):
    """Raised when scoring evidence is malformed or inconsistent."""


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be a finite number"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be a finite number"
        )
    return result


def _strict_float(value: Any, *, field: str, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be an exact finite Python float"
        )
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be an exact Python integer"
        )
    result = int(value)
    if result < minimum:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be at least {minimum}"
        )
    return result


def _exact_years(value: Any, *, field: str) -> tuple[int, ...]:
    try:
        years = tuple(value)
    except TypeError as exc:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be a sequence of exact Python integers"
        ) from exc
    if any(type(year) is not int for year in years):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must contain exact Python integers"
        )
    if tuple(sorted(set(years))) != years:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be unique and increasing"
        )
    return years


def _strict_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be an exact boolean"
        )
    return value


def _strict_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be a canonical SHA-256 identifier"
        )
    return value


def _canonical_sha256(value: Any, *, field: str) -> str:
    """Recompute a frozen artifact checksum without trusting ledger internals."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return item
        if isinstance(item, Real):
            number = float(item)
            if not math.isfinite(number):
                raise ContextualExpertAggregationEvaluationError(
                    f"{field} contains a nonfinite checksum value"
                )
            if isinstance(item, int) and not isinstance(item, bool):
                return int(item)
            return number
        raise ContextualExpertAggregationEvaluationError(
            f"{field} contains an unsupported checksum value"
        )

    try:
        payload = json.dumps(
            normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} cannot be represented canonically"
        ) from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_keys(value: Any, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} has missing or unexpected fields"
        )
    return value


def _frame_with_columns(
    value: pd.DataFrame,
    required: Sequence[str],
    *,
    field: str,
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} must be a pandas DataFrame"
        )
    expected = tuple(required)
    if tuple(value.columns) != expected:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} columns or column order differ from the frozen schema"
        )
    return value.copy()


def _positive_concentration(values: Sequence[float]) -> float | None:
    positive = [_finite(value, field="positive concentration value") for value in values]
    positive = [value for value in positive if value > 0.0]
    if not positive:
        return None
    return float(max(positive) / math.fsum(positive))


def _stats(values: Sequence[float], *, prefix: str) -> dict[str, Any]:
    clean = [_finite(value, field=f"{prefix} value") for value in values]
    if not clean:
        return {
            "count": 0,
            "positive_count": 0,
            "positive_rate": 0.0,
            "mean": None,
            "median": None,
            "positive_concentration": None,
            "sum": 0.0,
        }
    total = float(math.fsum(clean))
    count = len(clean)
    positive_count = sum(value > 0.0 for value in clean)
    ordered = sorted(clean)
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else math.fsum((ordered[middle - 1], ordered[middle])) / 2.0
    )
    return {
        "count": count,
        "positive_count": positive_count,
        "positive_rate": positive_count / count,
        "mean": total / count,
        "median": float(median),
        "positive_concentration": _positive_concentration(clean),
        "sum": total,
    }


def _cost_friction(cost_bps: float) -> float:
    cost = _strict_float(cost_bps, field="cost bps") / 10_000.0
    if cost < 0.0 or cost >= 1.0:
        raise ContextualExpertAggregationEvaluationError(
            "cost bps must imply a finite changing-leg multiplier"
        )
    return math.log((1.0 - cost) / (1.0 + cost))


def _date_column(frame: pd.DataFrame, name: str, *, field: str) -> pd.Series:
    raw = frame[name].tolist()
    for value in raw:
        if type(value) is not str:
            raise ContextualExpertAggregationEvaluationError(
                f"{field}.{name} must contain exact Python strings"
            )
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ContextualExpertAggregationEvaluationError(
                f"{field}.{name} must contain canonical YYYY-MM-DD dates"
            ) from exc
        if parsed.isoformat() != value:
            raise ContextualExpertAggregationEvaluationError(
                f"{field}.{name} must contain canonical YYYY-MM-DD dates"
            )
    try:
        dates = pd.to_datetime(frame[name], format="%Y-%m-%d", exact=True, errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.{name} must contain canonical YYYY-MM-DD dates"
        ) from exc
    return dates


def _numeric_column(
    frame: pd.DataFrame, name: str, *, field: str, positive: bool = False
) -> pd.Series:
    values = [
        _strict_float(value, field=f"{field}.{name}", positive=positive)
        for value in frame[name].tolist()
    ]
    return pd.Series(values, index=frame.index, name=name, dtype=float)


def _canonical_start_state(
    ledger: pd.DataFrame, *, cost_bps: float, field: str
) -> _ledger.AccountState:
    cost_value = _strict_float(cost_bps, field=f"{field} cost bps")
    if not isinstance(ledger, pd.DataFrame) or tuple(ledger.columns) != LEDGER_COLUMNS:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} does not have the canonical ledger schema"
        )
    if ledger.empty:
        raise ContextualExpertAggregationEvaluationError(f"{field} is empty")
    policies = ledger["policy_name"].tolist()
    if any(not isinstance(value, str) or value != policies[0] for value in policies):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} policy identity changes within the account"
        )
    observed_costs = ledger["cost_bps"].tolist()
    if any(
        type(value) is not float
        or not math.isfinite(value)
        or value != cost_value
        for value in observed_costs
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} cost scenario differs from its evidence key"
        )
    try:
        return _ledger.AccountState.initial(
            policy_name=policies[0], cost_bps=cost_value
        )
    except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} cannot construct its canonical initial state"
        ) from exc


def _validated_ledger(
    value: pd.DataFrame,
    *,
    field: str,
    cost_bps: float,
    account_years: Sequence[int],
) -> pd.DataFrame:
    start = _canonical_start_state(value, cost_bps=cost_bps, field=field)
    try:
        _ledger.verify_ledger(value, start_state=start)
        _ledger.assert_no_leverage(value)
    except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} failed canonical deterministic replay"
        ) from exc
    fills = pd.to_datetime(value["fill_date"], errors="raise")
    required_years = tuple(int(year) for year in account_years)
    if tuple(sorted(set(fills.dt.year.astype(int)))) != required_years:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} does not cover the exact account years"
        )
    return value.copy()


def _canonical_episode_frame(
    episodes: pd.DataFrame, *, cost_bps: float, field: str
) -> pd.DataFrame:
    cost_value = _strict_float(cost_bps, field=f"{field} cost bps")
    expected_cost = _cost_friction(cost_value)
    frame = _frame_with_columns(episodes, EPISODE_COLUMNS, field=field)
    if frame.empty:
        return frame
    entry_decision = _date_column(frame, "entry_decision_date", field=field)
    entry = _date_column(frame, "entry_fill_date", field=field)
    exit_decision = _date_column(frame, "exit_decision_date", field=field)
    exit_dates = _date_column(frame, "exit_fill_date", field=field)
    if (
        entry.duplicated().any()
        or not entry.is_monotonic_increasing
        or not (
            (entry_decision < entry)
            & (entry <= exit_decision)
            & (exit_decision < exit_dates)
        ).all()
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} entry/exit fills are duplicate or invalid"
        )
    if frame["episode_id"].duplicated().any():
        raise ContextualExpertAggregationEvaluationError(
            f"{field} contains duplicate episode identifiers"
        )
    for position, value in enumerate(frame["cash_fill_observations"].tolist()):
        _integer(
            value,
            field=f"{field}[{position}].cash_fill_observations",
            minimum=1,
        )
    for position, row in enumerate(frame.to_dict(orient="records")):
        _strict_sha256(row["episode_id"], field=f"{field}[{position}].episode_id")
        supplied_hash = _strict_sha256(
            row["episode_sha256"], field=f"{field}[{position}].episode_sha256"
        )
        payload = {name: row[name] for name in EPISODE_COLUMNS[:-1]}
        if supplied_hash != _canonical_sha256(
            payload, field=f"{field}[{position}]"
        ):
            raise ContextualExpertAggregationEvaluationError(
                f"{field}[{position}] episode checksum does not match its row"
            )
    entry_reference = _numeric_column(
        frame, "entry_reference_price", field=field, positive=True
    )
    entry_fill = _numeric_column(
        frame, "entry_sell_fill_price", field=field, positive=True
    )
    exit_reference = _numeric_column(
        frame, "exit_reference_price", field=field, positive=True
    )
    exit_fill = _numeric_column(
        frame, "exit_buy_fill_price", field=field, positive=True
    )
    raw = _numeric_column(frame, "raw_active_log_edge", field=field)
    cost_edge = _numeric_column(frame, "cost_log_edge", field=field)
    net = _numeric_column(frame, "net_active_log_edge", field=field)
    cost = cost_value / 10_000.0
    if (
        not np.allclose(
            entry_fill.to_numpy(dtype=float),
            entry_reference.to_numpy(dtype=float) * (1.0 - cost),
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            exit_fill.to_numpy(dtype=float),
            exit_reference.to_numpy(dtype=float) * (1.0 + cost),
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            raw.to_numpy(dtype=float),
            np.log(
                entry_reference.to_numpy(dtype=float)
                / exit_reference.to_numpy(dtype=float)
            ),
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            cost_edge.to_numpy(dtype=float),
            expected_cost,
            rtol=0.0,
            atol=1e-15,
        )
        or not np.allclose(
            net.to_numpy(dtype=float),
            raw.to_numpy(dtype=float) + expected_cost,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} changing-leg costs or net edge changed"
        )
    frame["raw_active_log_edge"] = raw
    frame["cost_log_edge"] = cost_edge
    frame["net_active_log_edge"] = net
    return frame


def summarize_complete_episodes(
    episodes: pd.DataFrame,
    *,
    years: Sequence[int],
    cost_bps: float,
) -> dict[str, Any]:
    frame = _canonical_episode_frame(
        episodes, cost_bps=cost_bps, field="episodes"
    )
    selected_years = _exact_years(years, field="episode reporting years")
    entry_year = (
        _date_column(frame, "entry_fill_date", field="episodes").dt.year
        if not frame.empty
        else pd.Series(dtype=int)
    )
    selected = frame.loc[entry_year.isin(selected_years)]
    selected_entry_year = entry_year.loc[selected.index]
    result = _stats(selected["net_active_log_edge"].tolist(), prefix="episode")
    result["year_edges"] = {
        str(year): float(
            math.fsum(
                selected.loc[selected_entry_year == year, "net_active_log_edge"].tolist()
            )
        )
        for year in selected_years
    }
    return result


def _canonical_xor_frame(
    differences: pd.DataFrame, *, cost_bps: float, field: str
) -> pd.DataFrame:
    cost_value = _strict_float(cost_bps, field=f"{field} cost bps")
    _cost_friction(cost_value)
    frame = _frame_with_columns(differences, XOR_COLUMNS, field=field)
    if frame.empty:
        return frame
    entry = _date_column(frame, "entry_fill_date", field=field)
    exit_dates = _date_column(frame, "exit_fill_date", field=field)
    if (
        entry.duplicated().any()
        or not entry.is_monotonic_increasing
        or not (entry < exit_dates).all()
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} entry/exit fills are duplicate or invalid"
        )
    if not frame["orientation"].isin(XOR_ORIENTATIONS).all():
        raise ContextualExpertAggregationEvaluationError(
            f"{field} contains a mixed or invalid contract orientation"
        )
    if frame["xor_id"].duplicated().any():
        raise ContextualExpertAggregationEvaluationError(
            f"{field} contains duplicate XOR identifiers"
        )
    for position, value in enumerate(frame["xor_fill_observations"].tolist()):
        _integer(
            value,
            field=f"{field}[{position}].xor_fill_observations",
            minimum=1,
        )
    for position, row in enumerate(frame.to_dict(orient="records")):
        _strict_sha256(row["xor_id"], field=f"{field}[{position}].xor_id")
        supplied_hash = _strict_sha256(
            row["xor_sha256"], field=f"{field}[{position}].xor_sha256"
        )
        payload = {name: row[name] for name in XOR_COLUMNS[:-1]}
        if supplied_hash != _canonical_sha256(
            payload, field=f"{field}[{position}]"
        ):
            raise ContextualExpertAggregationEvaluationError(
                f"{field}[{position}] XOR checksum does not match its row"
            )
    for name in (
        "raw_market_component",
        "transition_cost_component",
        "net_signed_log_edge",
    ):
        frame[name] = _numeric_column(frame, name, field=field)
    if not np.allclose(
        frame["net_signed_log_edge"].to_numpy(dtype=float),
        frame["raw_market_component"].to_numpy(dtype=float)
        + frame["transition_cost_component"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} net signed edges do not equal raw plus transition cost"
        )
    return frame


def summarize_xor_differences(
    differences: pd.DataFrame,
    *,
    years: Sequence[int],
    cost_bps: float,
) -> dict[str, Any]:
    frame = _canonical_xor_frame(
        differences, cost_bps=cost_bps, field="XOR differences"
    )
    selected_years = _exact_years(years, field="XOR reporting years")
    entry_year = (
        _date_column(frame, "entry_fill_date", field="XOR differences").dt.year
        if not frame.empty
        else pd.Series(dtype=int)
    )
    selected = frame.loc[entry_year.isin(selected_years)]
    selected_entry_year = entry_year.loc[selected.index]
    result = _stats(selected["net_signed_log_edge"].tolist(), prefix="XOR")
    result["year_edges"] = {
        str(year): float(
            math.fsum(
                selected.loc[
                    selected_entry_year == year, "net_signed_log_edge"
                ].tolist()
            )
        )
        for year in selected_years
    }
    result["positive_year_count"] = int(
        sum(value > 0.0 for value in result["year_edges"].values())
    )
    result["orientation_values"] = sorted(set(selected["orientation"].tolist()))
    return result


def summarize_policy_ledgers(
    strategy_ledger: pd.DataFrame,
    benchmark_ledger: pd.DataFrame,
    complete_episodes: pd.DataFrame,
    *,
    reporting_years: Sequence[int],
    account_years: Sequence[int],
    cost_bps: float,
    folds: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    strategy = _validated_ledger(
        strategy_ledger,
        field="strategy ledger",
        cost_bps=cost_bps,
        account_years=account_years,
    )
    benchmark = _validated_ledger(
        benchmark_ledger,
        field="benchmark ledger",
        cost_bps=cost_bps,
        account_years=account_years,
    )
    strategy_start = _canonical_start_state(
        strategy, cost_bps=cost_bps, field="strategy ledger"
    )
    benchmark_start = _canonical_start_state(
        benchmark, cost_bps=cost_bps, field="benchmark ledger"
    )
    if not (benchmark["requested_target_exposure"] == 1).all():
        raise ContextualExpertAggregationEvaluationError(
            "same-ledger benchmark is not always LONG"
        )
    try:
        extracted = _ledger.extract_cash_episodes(
            strategy, start_state=strategy_start
        )
        supplied = _canonical_episode_frame(
            complete_episodes, cost_bps=cost_bps, field="complete episodes"
        )
        if not extracted.complete.equals(supplied):
            raise ContextualExpertAggregationEvaluationError(
                "complete episode artifact differs from canonical ledger extraction"
            )
        reconciliation = _ledger.reconcile_complete_cash_episodes(
            strategy,
            benchmark,
            extracted,
            strategy_start_state=strategy_start,
            buy_hold_start_state=benchmark_start,
        )
    except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            "canonical ledger/episode reconciliation failed"
        ) from exc
    years = _exact_years(reporting_years, field="policy reporting years")
    if not set(years).issubset(set(int(year) for year in account_years)):
        raise ContextualExpertAggregationEvaluationError(
            "reporting years are not an exact account subset"
        )
    episode_summary = summarize_complete_episodes(
        supplied, years=years, cost_bps=cost_bps
    )
    all_years = tuple(
        sorted(
            set(
                _date_column(
                    supplied, "entry_fill_date", field="complete episodes"
                ).dt.year.tolist()
            )
        )
    )
    all_episode_summary = summarize_complete_episodes(
        supplied, years=all_years, cost_bps=cost_bps
    )
    strategy_returns = strategy["daily_return"].to_numpy(dtype=float)
    benchmark_returns = benchmark["daily_return"].to_numpy(dtype=float)
    active = np.log1p(strategy_returns) - np.log1p(benchmark_returns)
    fill_years = pd.to_datetime(strategy["fill_date"]).dt.year.to_numpy(dtype=int)
    annual: dict[str, Any] = {}
    for year in years:
        mask = fill_years == year
        if not np.any(mask):
            raise ContextualExpertAggregationEvaluationError(
                f"continuous ledger has no fills in reporting year {year}"
            )
        benchmark_log_return = float(
            math.fsum(
                math.log1p(float(value))
                for value in benchmark_returns[mask].tolist()
            )
        )
        annual[str(year)] = {
            "entry_attributed_active_log_edge": episode_summary["year_edges"][str(year)],
            "ledger_boundary_active_log_edge": float(math.fsum(active[mask].tolist())),
            "aapl_buy_hold_return": float(math.expm1(benchmark_log_return)),
        }
    fold_edges: dict[str, float] = {}
    for name, fold_years in (folds or {}).items():
        values = _exact_years(fold_years, field=f"fold {name} years")
        if not values or not set(values).issubset(set(years)):
            raise ContextualExpertAggregationEvaluationError(
                "fold years are not an exact reporting subset"
            )
        fold_edges[name] = float(
            math.fsum(episode_summary["year_edges"][str(year)] for year in values)
        )
    negative_years = [
        year for year in years if annual[str(year)]["aapl_buy_hold_return"] < 0.0
    ]
    negative_edges = {
        str(year): annual[str(year)]["entry_attributed_active_log_edge"]
        for year in negative_years
    }
    reporting_edge = float(episode_summary["sum"])
    terminal_equity_edge = float(reconciliation["ledger_active_log_edge"])
    attributed_full = float(all_episode_summary["sum"])
    error = float(math.fsum((terminal_equity_edge, -attributed_full)))
    return {
        # Strict economic gates use the accurately summed atomic episode edge.
        # Terminal-equity log arithmetic remains an independent reconciliation
        # diagnostic and may differ by a few floating-point ulps.
        "full_account_active_log_edge": attributed_full,
        "entry_attributed_full_account_active_log_edge": attributed_full,
        "full_account_episode_reconciliation_error": error,
        "full_account_episode_reconciled": abs(error) <= RECONCILIATION_TOLERANCE,
        "reporting_active_log_edge": reporting_edge,
        "reporting_ledger_boundary_active_log_edge": float(
            math.fsum(annual[str(year)]["ledger_boundary_active_log_edge"] for year in years)
        ),
        "annual": annual,
        "fold_edges": fold_edges,
        "positive_year_count": sum(
            annual[str(year)]["entry_attributed_active_log_edge"] > 0.0 for year in years
        ),
        "positive_fold_count": sum(value > 0.0 for value in fold_edges.values()),
        "edge_after_removing_best_year": float(
            math.fsum(
                (
                    reporting_edge,
                    -max(episode_summary["year_edges"].values()),
                )
            )
        ),
        "edge_after_removing_best_fold": (
            math.fsum(
                (math.fsum(fold_edges.values()), -max(fold_edges.values()))
            )
            if fold_edges
            else 0.0
        ),
        "negative_aapl_years": negative_years,
        "negative_aapl_year_edges": negative_edges,
        "negative_aapl_year_edge_sum": float(math.fsum(negative_edges.values())),
        "episodes": episode_summary,
    }


def _validated_policy_summary(value: Any, *, field: str) -> Mapping[str, Any]:
    summary = _strict_keys(
        value,
        {
            "full_account_active_log_edge",
            "entry_attributed_full_account_active_log_edge",
            "full_account_episode_reconciliation_error",
            "full_account_episode_reconciled",
            "reporting_active_log_edge",
            "reporting_ledger_boundary_active_log_edge",
            "annual",
            "fold_edges",
            "positive_year_count",
            "positive_fold_count",
            "edge_after_removing_best_year",
            "edge_after_removing_best_fold",
            "negative_aapl_years",
            "negative_aapl_year_edges",
            "negative_aapl_year_edge_sum",
            "episodes",
        },
        field=field,
    )
    for name in (
        "full_account_active_log_edge",
        "entry_attributed_full_account_active_log_edge",
        "full_account_episode_reconciliation_error",
        "reporting_active_log_edge",
        "reporting_ledger_boundary_active_log_edge",
        "edge_after_removing_best_year",
        "edge_after_removing_best_fold",
        "negative_aapl_year_edge_sum",
    ):
        _finite(summary[name], field=f"{field}.{name}")
    _strict_bool(
        summary["full_account_episode_reconciled"],
        field=f"{field}.full_account_episode_reconciled",
    )
    _integer(summary["positive_year_count"], field=f"{field}.positive_year_count")
    _integer(summary["positive_fold_count"], field=f"{field}.positive_fold_count")
    if not isinstance(summary["annual"], Mapping) or not isinstance(
        summary["fold_edges"], Mapping
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} annual/fold metrics must be mappings"
        )
    if not isinstance(summary["negative_aapl_years"], list) or not isinstance(
        summary["negative_aapl_year_edges"], Mapping
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} negative-year metrics are malformed"
        )
    episodes = summary["episodes"]
    if not isinstance(episodes, Mapping):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.episodes must be a mapping"
        )
    for name in ("count", "positive_count"):
        _integer(episodes.get(name), field=f"{field}.episodes.{name}")
    if set(episodes) != {
        "count",
        "positive_count",
        "positive_rate",
        "mean",
        "median",
        "positive_concentration",
        "sum",
        "year_edges",
    }:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.episodes has missing or unexpected fields"
        )
    for name in ("positive_rate", "sum"):
        _finite(episodes.get(name), field=f"{field}.episodes.{name}")
    for name in ("mean", "median", "positive_concentration"):
        if episodes.get(name) is not None:
            _finite(episodes[name], field=f"{field}.episodes.{name}")
    year_edges = episodes.get("year_edges")
    if not isinstance(year_edges, Mapping):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.episodes.year_edges must be a mapping"
        )
    for year, edge in year_edges.items():
        if not isinstance(year, str):
            raise ContextualExpertAggregationEvaluationError(
                f"{field}.episodes.year_edges keys must be strings"
            )
        _finite(edge, field=f"{field}.episodes.year_edges[{year}]")
    return summary


def _require_summary_period(
    summary: Mapping[str, Any],
    *,
    years: Sequence[int],
    fold_names: set[str],
    field: str,
) -> None:
    """Recompute all period-level fields consumed by gates."""

    expected_year_keys = {str(int(year)) for year in years}
    annual = summary["annual"]
    if set(annual) != expected_year_keys:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.annual does not cover the exact frozen years"
        )
    annual_edges: dict[str, float] = {}
    annual_ledger_edges: dict[str, float] = {}
    negative_years: list[int] = []
    for year in years:
        key = str(int(year))
        row = _strict_keys(
            annual[key],
            {
                "entry_attributed_active_log_edge",
                "ledger_boundary_active_log_edge",
                "aapl_buy_hold_return",
            },
            field=f"{field}.annual[{key}]",
        )
        annual_edges[key] = _finite(
            row["entry_attributed_active_log_edge"],
            field=f"{field}.annual[{key}].entry edge",
        )
        annual_ledger_edges[key] = _finite(
            row["ledger_boundary_active_log_edge"],
            field=f"{field}.annual[{key}].ledger edge",
        )
        benchmark_return = _finite(
            row["aapl_buy_hold_return"],
            field=f"{field}.annual[{key}].AAPL return",
        )
        if benchmark_return < 0.0:
            negative_years.append(int(year))

    reporting_edge = _finite(
        summary["reporting_active_log_edge"], field=f"{field}.reporting edge"
    )
    if not math.isclose(
        reporting_edge,
        math.fsum(annual_edges.values()),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.reporting edge does not equal entry-attributed annual edges"
        )
    if not math.isclose(
        _finite(
            summary["reporting_ledger_boundary_active_log_edge"],
            field=f"{field}.reporting ledger edge",
        ),
        math.fsum(annual_ledger_edges.values()),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.reporting ledger edge is not recomputable"
        )
    expected_positive_years = sum(value > 0.0 for value in annual_edges.values())
    if summary["positive_year_count"] != expected_positive_years:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.positive year count is not recomputable"
        )
    expected_after_best_year = math.fsum(
        (reporting_edge, -max(annual_edges.values()))
    )
    if not math.isclose(
        _finite(
            summary["edge_after_removing_best_year"],
            field=f"{field}.edge after best year",
        ),
        expected_after_best_year,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.best-year removal is not recomputable"
        )

    folds = summary["fold_edges"]
    if set(folds) != fold_names:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.fold inventory is not exact"
        )
    clean_folds = {
        name: _finite(edge, field=f"{field}.fold_edges[{name}]")
        for name, edge in folds.items()
    }
    if clean_folds:
        if not math.isclose(
            math.fsum(clean_folds.values()),
            reporting_edge,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ContextualExpertAggregationEvaluationError(
                f"{field}.fold edges do not cover the reporting edge"
            )
        expected_after_best_fold = math.fsum(
            (math.fsum(clean_folds.values()), -max(clean_folds.values()))
        )
    else:
        expected_after_best_fold = 0.0
    if summary["positive_fold_count"] != sum(
        value > 0.0 for value in clean_folds.values()
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.positive fold count is not recomputable"
        )
    if not math.isclose(
        _finite(
            summary["edge_after_removing_best_fold"],
            field=f"{field}.edge after best fold",
        ),
        expected_after_best_fold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.best-fold removal is not recomputable"
        )

    if summary["negative_aapl_years"] != negative_years:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.negative AAPL years are not mechanically selected"
        )
    expected_negative_edges = {
        str(year): annual_edges[str(year)] for year in negative_years
    }
    observed_negative_edges = {
        str(year): _finite(edge, field=f"{field}.negative year edge")
        for year, edge in summary["negative_aapl_year_edges"].items()
    }
    if observed_negative_edges != expected_negative_edges:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.negative-year edges are not entry-attributed"
        )
    if not math.isclose(
        _finite(
            summary["negative_aapl_year_edge_sum"],
            field=f"{field}.negative-year edge sum",
        ),
        math.fsum(expected_negative_edges.values()),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.negative-year edge sum is not recomputable"
        )

    episodes = summary["episodes"]
    if set(episodes["year_edges"]) != expected_year_keys or any(
        not math.isclose(
            _finite(episodes["year_edges"][key], field="episode year edge"),
            annual_edges[key],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for key in expected_year_keys
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.episode year edges differ from annual edges"
        )
    if not math.isclose(
        _finite(episodes["sum"], field=f"{field}.episode sum"),
        reporting_edge,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.episode sum differs from reporting edge"
        )
    episode_count = _integer(episodes["count"], field=f"{field}.episode count")
    positive_count = _integer(
        episodes["positive_count"], field=f"{field}.episode positive count"
    )
    if positive_count > episode_count:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.episode positive count exceeds total"
        )
    expected_rate = positive_count / episode_count if episode_count else 0.0
    if not math.isclose(
        _finite(episodes["positive_rate"], field=f"{field}.episode rate"),
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.episode positive rate is not recomputable"
        )
    if episode_count:
        if episodes["mean"] is None or not math.isclose(
            _finite(episodes["mean"], field=f"{field}.episode mean")
            * episode_count,
            reporting_edge,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ContextualExpertAggregationEvaluationError(
                f"{field}.episode mean is not recomputable"
            )
    elif (
        episodes["mean"] is not None
        or episodes["median"] is not None
        or episodes["positive_concentration"] is not None
        or reporting_edge != 0.0
        or positive_count != 0
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.empty episode summary is inconsistent"
        )
    concentration = episodes["positive_concentration"]
    if concentration is not None and not 0.0 <= _finite(
        concentration, field=f"{field}.episode concentration"
    ) <= 1.0:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.episode concentration is outside [0,1]"
        )
    full_edge = _finite(
        summary["full_account_active_log_edge"], field=f"{field}.full edge"
    )
    attributed_full_edge = _finite(
        summary["entry_attributed_full_account_active_log_edge"],
        field=f"{field}.attributed full edge",
    )
    supplied_error = _finite(
        summary["full_account_episode_reconciliation_error"],
        field=f"{field}.full reconciliation error",
    )
    expected_error = math.fsum((full_edge, -attributed_full_edge))
    if not math.isclose(
        supplied_error, expected_error, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.full-account reconciliation error is not recomputable"
        )
    expected_reconciled = abs(expected_error) <= RECONCILIATION_TOLERANCE
    if summary["full_account_episode_reconciled"] is not expected_reconciled:
        raise ContextualExpertAggregationEvaluationError(
            f"{field}.full-account reconciliation flag is inconsistent"
        )


def _integrity_checks(
    value: Mapping[str, Any], *, expected: frozenset[str], stage: str
) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ContextualExpertAggregationEvaluationError(
            f"{stage} integrity evidence does not have the exact frozen inventory"
        )
    return {
        f"integrity.{name}": _strict_bool(flag, field=f"integrity.{name}")
        for name, flag in sorted(value.items())
    }


def _gate_report(checks: Mapping[str, bool]) -> dict[str, Any]:
    if not checks:
        raise ContextualExpertAggregationEvaluationError(
            "gate report requires at least one check"
        )
    normalized = {
        name: _strict_bool(value, field=f"gate.{name}")
        for name, value in sorted(checks.items())
    }
    return {
        "passed": bool(all(normalized.values())),
        "passed_count": int(sum(normalized.values())),
        "total_count": len(normalized),
        "checks": normalized,
        "failed_checks": [name for name, passed in normalized.items() if not passed],
    }


def _require_frame_equal(left: pd.DataFrame, right: pd.DataFrame, *, field: str) -> None:
    if tuple(left.columns) != tuple(right.columns) or not left.equals(right):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} differs despite requiring exact same-ledger evidence"
        )


def _require_shared_account_prefix(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    cutoff_fill_date: str,
    field: str,
) -> None:
    ignored = {"policy_name", "previous_row_sha256", "row_sha256"}
    compared = [name for name in LEDGER_COLUMNS if name not in ignored]
    primary_prefix = primary.loc[
        primary["fill_date"] <= cutoff_fill_date, compared
    ].reset_index(drop=True)
    comparator_prefix = comparator.loc[
        comparator["fill_date"] <= cutoff_fill_date, compared
    ].reset_index(drop=True)
    if primary_prefix.empty or not primary_prefix.equals(comparator_prefix):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} does not share the exact through-2018 account/action prefix"
        )


def _require_same_action_stream(
    left: pd.DataFrame, right: pd.DataFrame, *, field: str
) -> None:
    try:
        _ledger.assert_cross_cost_action_identity({"base": left, "stress": right})
    except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} action stream changed between cost scenarios"
        ) from exc


def _require_episode_identity(
    left: pd.DataFrame, right: pd.DataFrame, *, field: str
) -> None:
    identity = (
        "episode_id",
        "entry_decision_date",
        "entry_fill_date",
        "exit_decision_date",
        "exit_fill_date",
        "entry_reference_price",
        "exit_reference_price",
        "cash_fill_observations",
        "raw_active_log_edge",
    )
    if len(left) != len(right) or any(
        not np.array_equal(left[name].to_numpy(), right[name].to_numpy())
        for name in identity
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} episode identity changed between cost scenarios"
        )


def _require_xor_identity(
    left: pd.DataFrame, right: pd.DataFrame, *, field: str
) -> None:
    identity = (
        "xor_id",
        "entry_fill_date",
        "exit_fill_date",
        "orientation",
        "xor_fill_observations",
        "raw_market_component",
    )
    if len(left) != len(right) or any(
        not np.array_equal(left[name].to_numpy(), right[name].to_numpy())
        for name in identity
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} XOR identity changed between cost scenarios"
        )


def _stage_spec(stage: str) -> tuple[tuple[int, ...], tuple[int, ...], Mapping[str, Sequence[int]] | None]:
    if stage == "development":
        return DEVELOPMENT_YEARS, DEVELOPMENT_YEARS, DEVELOPMENT_FOLDS
    if stage == "confirmation":
        return CONFIRMATION_YEARS, CONFIRMATION_ACCOUNT_YEARS, None
    raise ContextualExpertAggregationEvaluationError(f"unknown evaluation stage: {stage}")


def _evaluate_policy_evidence(
    evidence: Any, *, cost_name: str, stage: str, field: str
) -> dict[str, Any]:
    value = _strict_keys(evidence, set(POLICY_EVIDENCE_KEYS), field=field)
    reporting_years, account_years, folds = _stage_spec(stage)
    cost_bps = COST_BPS[cost_name]
    summary = summarize_policy_ledgers(
        value["strategy_ledger"],
        value["benchmark_ledger"],
        value["complete_episodes"],
        reporting_years=reporting_years,
        account_years=account_years,
        cost_bps=cost_bps,
        folds=folds,
    )
    _require_summary_period(
        _validated_policy_summary(summary, field=f"{field}.summary"),
        years=reporting_years,
        fold_names=set(folds or {}),
        field=f"{field}.summary",
    )
    return {
        "summary": summary,
        "strategy_ledger": _validated_ledger(
            value["strategy_ledger"],
            field=f"{field}.strategy",
            cost_bps=cost_bps,
            account_years=account_years,
        ),
        "benchmark_ledger": _validated_ledger(
            value["benchmark_ledger"],
            field=f"{field}.benchmark",
            cost_bps=cost_bps,
            account_years=account_years,
        ),
        "episodes": _canonical_episode_frame(
            value["complete_episodes"], cost_bps=cost_bps, field=f"{field}.episodes"
        ),
    }


def _evaluate_primary_by_cost(
    evidence: Any, *, stage: str
) -> dict[str, dict[str, Any]]:
    by_cost = _strict_keys(evidence, set(COST_NAMES), field=f"{stage} learner evidence")
    result = {
        cost_name: _evaluate_policy_evidence(
            by_cost[cost_name],
            cost_name=cost_name,
            stage=stage,
            field=f"{stage}.learner.{cost_name}",
        )
        for cost_name in COST_NAMES
    }
    _require_same_action_stream(
        result[BASE_COST_NAME]["strategy_ledger"],
        result[STRESS_COST_NAME]["strategy_ledger"],
        field=f"{stage} learner",
    )
    _require_same_action_stream(
        result[BASE_COST_NAME]["benchmark_ledger"],
        result[STRESS_COST_NAME]["benchmark_ledger"],
        field=f"{stage} benchmark",
    )
    _require_episode_identity(
        result[BASE_COST_NAME]["episodes"],
        result[STRESS_COST_NAME]["episodes"],
        field=f"{stage} learner",
    )
    return result


def _require_xor_ledger_identity(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    differences: pd.DataFrame,
    *,
    cost_bps: float,
    field: str,
) -> None:
    primary_start = _canonical_start_state(
        primary, cost_bps=cost_bps, field=f"{field}.primary"
    )
    comparator_start = _canonical_start_state(
        comparator, cost_bps=cost_bps, field=f"{field}.comparator"
    )
    try:
        extraction = _ledger.extract_signed_xor_episodes(
            primary,
            comparator,
            primary_start_state=primary_start,
            comparator_start_state=comparator_start,
        )
        if not extraction.unresolved.empty:
            raise ContextualExpertAggregationEvaluationError(
                f"{field} has a partial XOR boundary"
            )
        if not extraction.complete.equals(differences):
            raise ContextualExpertAggregationEvaluationError(
                f"{field} XOR artifact differs from canonical maximal-run extraction"
            )
        _ledger.reconcile_signed_xor(
            primary,
            comparator,
            extraction,
            primary_start_state=primary_start,
            comparator_start_state=comparator_start,
        )
    except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            f"{field} canonical XOR replay or reconciliation failed"
        ) from exc


def _evaluate_pairwise_evidence(
    evidence: Any,
    *,
    primary: Mapping[str, Any],
    cost_name: str,
    stage: str,
    field: str,
) -> dict[str, Any]:
    value = _strict_keys(evidence, set(PAIRWISE_EVIDENCE_KEYS), field=field)
    policy = _evaluate_policy_evidence(
        {name: value[name] for name in POLICY_EVIDENCE_KEYS},
        cost_name=cost_name,
        stage=stage,
        field=f"{field}.comparator",
    )
    _require_frame_equal(
        primary["benchmark_ledger"],
        policy["benchmark_ledger"],
        field=f"{field} benchmark",
    )
    cost_bps = COST_BPS[cost_name]
    xor = _canonical_xor_frame(
        value["learner_minus_comparator_xor"],
        cost_bps=cost_bps,
        field=f"{field}.xor",
    )
    _require_xor_ledger_identity(
        primary["strategy_ledger"],
        policy["strategy_ledger"],
        xor,
        cost_bps=cost_bps,
        field=field,
    )
    reporting_years, account_years, _ = _stage_spec(stage)
    selected = summarize_xor_differences(
        xor, years=reporting_years, cost_bps=cost_bps
    )
    all_years = summarize_xor_differences(
        xor, years=account_years, cost_bps=cost_bps
    )
    primary_summary = primary["summary"]
    comparator_summary = policy["summary"]
    expected_full = float(
        math.fsum(
            (
                primary_summary["full_account_active_log_edge"],
                -comparator_summary["full_account_active_log_edge"],
            )
        )
    )
    if (
        abs(math.fsum((expected_full, -float(all_years["sum"]))))
        > RECONCILIATION_TOLERANCE
    ):
        raise ContextualExpertAggregationEvaluationError(
            f"{field} aggregate XOR edge does not reconcile to policy ledgers"
        )
    return {
        "policy": policy,
        "xor": xor,
        "xor_summary": selected,
        "full_xor_summary": all_years,
        "full_incremental_edge": expected_full,
        "reporting_incremental_edge": float(selected["sum"]),
    }


def _evaluate_pairwise_inventory(
    evidence: Any,
    *,
    names: Sequence[str],
    primary_by_cost: Mapping[str, Mapping[str, Any]],
    stage: str,
    field: str,
) -> dict[str, dict[str, Any]]:
    by_cost = _strict_keys(evidence, set(COST_NAMES), field=field)
    result: dict[str, dict[str, Any]] = {name: {} for name in names}
    for cost_name in COST_NAMES:
        items = _strict_keys(
            by_cost[cost_name], set(names), field=f"{field}.{cost_name}"
        )
        for name in names:
            result[name][cost_name] = _evaluate_pairwise_evidence(
                items[name],
                primary=primary_by_cost[cost_name],
                cost_name=cost_name,
                stage=stage,
                field=f"{field}.{cost_name}.{name}",
            )
    for name in names:
        base = result[name][BASE_COST_NAME]
        stress = result[name][STRESS_COST_NAME]
        _require_same_action_stream(
            base["policy"]["strategy_ledger"],
            stress["policy"]["strategy_ledger"],
            field=f"{field}.{name}",
        )
        _require_episode_identity(
            base["policy"]["episodes"],
            stress["policy"]["episodes"],
            field=f"{field}.{name}",
        )
        _require_xor_identity(
            base["xor"], stress["xor"], field=f"{field}.{name}"
        )
    return result


def apply_development_gates(
    policy_evidence: Mapping[str, Mapping[str, Any]],
    *,
    fixed_comparator_evidence: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay raw evidence and apply the frozen 2005-2018 gates."""

    primary = _evaluate_primary_by_cost(policy_evidence, stage="development")
    fixed = _evaluate_pairwise_inventory(
        fixed_comparator_evidence,
        names=FIXED_COMPARATOR_NAMES,
        primary_by_cost=primary,
        stage="development",
        field="development fixed comparators",
    )
    checks = _integrity_checks(
        integrity,
        expected=DEVELOPMENT_INTEGRITY_CHECKS,
        stage="development",
    )
    for cost_name in COST_NAMES:
        summary = primary[cost_name]["summary"]
        episodes = summary["episodes"]
        concentration = episodes["positive_concentration"]
        negative_years = summary["negative_aapl_years"]
        prefix = cost_name
        checks.update(
            {
                f"{prefix}.active_edge_gt_0_001": _finite(
                    summary["reporting_active_log_edge"], field="reporting edge"
                )
                > MIN_ACTIVE_LOG_EDGE,
                f"{prefix}.positive_folds_at_least_5": _integer(
                    summary["positive_fold_count"], field="positive fold count"
                )
                >= 5,
                f"{prefix}.edge_after_best_fold_positive": _finite(
                    summary["edge_after_removing_best_fold"],
                    field="edge after best fold",
                )
                > 0.0,
                f"{prefix}.episodes_at_least_30": _integer(
                    episodes["count"], field="episode count"
                )
                >= 30,
                f"{prefix}.episode_win_rate_at_least_55pct": _finite(
                    episodes["positive_rate"], field="episode win rate"
                )
                >= 0.55,
                f"{prefix}.episode_mean_positive": episodes["mean"] is not None
                and _finite(episodes["mean"], field="episode mean") > 0.0,
                f"{prefix}.episode_median_positive": episodes["median"] is not None
                and _finite(episodes["median"], field="episode median") > 0.0,
                f"{prefix}.positive_episode_concentration_at_most_50pct": (
                    concentration is not None
                    and _finite(concentration, field="episode concentration") <= 0.5
                ),
                f"{prefix}.negative_aapl_year_edge_positive": _finite(
                    summary["negative_aapl_year_edge_sum"],
                    field="negative-year edge",
                )
                > 0.0,
                f"{prefix}.full_account_episode_reconciled": summary[
                    "full_account_episode_reconciled"
                ]
                is True,
                f"{prefix}.development_reporting_equals_full_account": math.isclose(
                    _finite(
                        summary["reporting_active_log_edge"],
                        field="development reporting edge",
                    ),
                    _finite(
                        summary["full_account_active_log_edge"],
                        field="development full-account edge",
                    ),
                    rel_tol=0.0,
                    abs_tol=RECONCILIATION_TOLERANCE,
                ),
            }
        )
    learner_stress = _finite(
        primary[STRESS_COST_NAME]["summary"]["full_account_active_log_edge"],
        field="development stress full-account edge",
    )
    comparator_values = {
        name: _finite(
            fixed[name][STRESS_COST_NAME]["policy"]["summary"][
                "full_account_active_log_edge"
            ],
            field=f"development comparator {name}",
        )
        for name in FIXED_COMPARATOR_NAMES
    }
    always_long = fixed["always_long"][STRESS_COST_NAME]["policy"]
    try:
        _ledger.assert_always_long_matches_buy_hold(
            always_long["strategy_ledger"], always_long["benchmark_ledger"]
        )
    except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
        raise ContextualExpertAggregationEvaluationError(
            "development always-LONG control differs from buy-and-hold"
        ) from exc
    best_name = max(comparator_values, key=comparator_values.__getitem__)
    best_edge = comparator_values[best_name]
    learner_minus_best = math.fsum((learner_stress, -best_edge))
    checks["stress_10bps.full_account_beats_best_fixed_by_gt_0_0001"] = (
        learner_minus_best > STRICT_INCREMENTAL_EDGE
    )
    report = _gate_report(checks)
    report["best_fixed_comparator"] = {
        "name": best_name,
        "active_log_edge": best_edge,
        "learner_minus_comparator": learner_minus_best,
    }
    report["policy_summaries"] = {
        cost: primary[cost]["summary"] for cost in COST_NAMES
    }
    report["fixed_comparator_differences"] = {
        cost: {
            name: fixed[name][cost]["full_incremental_edge"]
            for name in FIXED_COMPARATOR_NAMES
        }
        for cost in COST_NAMES
    }
    return report


def apply_confirmation_gates(
    policy_evidence: Mapping[str, Mapping[str, Any]],
    *,
    fixed_comparator_evidence: Mapping[str, Mapping[str, Any]],
    ablation_evidence: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay raw evidence and apply the frozen 2019-2023 gates."""

    primary = _evaluate_primary_by_cost(policy_evidence, stage="confirmation")
    fixed = _evaluate_pairwise_inventory(
        fixed_comparator_evidence,
        names=FIXED_COMPARATOR_NAMES,
        primary_by_cost=primary,
        stage="confirmation",
        field="confirmation fixed comparators",
    )
    ablations = _evaluate_pairwise_inventory(
        ablation_evidence,
        names=ABLATION_COMPARISON_NAMES,
        primary_by_cost=primary,
        stage="confirmation",
        field="confirmation ablations",
    )
    for comparison_name in ABLATION_COMPARISON_NAMES:
        for cost_name in COST_NAMES:
            _require_shared_account_prefix(
                primary[cost_name]["strategy_ledger"],
                ablations[comparison_name][cost_name]["policy"]["strategy_ledger"],
                cutoff_fill_date="2018-12-31",
                field=f"confirmation ablation {comparison_name}.{cost_name}",
            )
    checks = _integrity_checks(
        integrity,
        expected=CONFIRMATION_INTEGRITY_CHECKS,
        stage="confirmation",
    )
    comparator_diagnostics: dict[str, Any] = {}

    for cost_name in COST_NAMES:
        summary = primary[cost_name]["summary"]
        episodes = summary["episodes"]
        concentration = episodes["positive_concentration"]
        prefix = cost_name
        checks.update(
            {
                f"{prefix}.active_edge_gt_0_001": _finite(
                    summary["reporting_active_log_edge"], field="reporting edge"
                )
                > MIN_ACTIVE_LOG_EDGE,
                f"{prefix}.positive_years_at_least_3": _integer(
                    summary["positive_year_count"], field="positive year count"
                )
                >= 3,
                f"{prefix}.edge_after_best_year_positive": _finite(
                    summary["edge_after_removing_best_year"],
                    field="edge after best year",
                )
                > 0.0,
                f"{prefix}.episodes_at_least_5": _integer(
                    episodes["count"], field="episode count"
                )
                >= 5,
                f"{prefix}.episode_mean_positive": episodes["mean"] is not None
                and _finite(episodes["mean"], field="episode mean") > 0.0,
                f"{prefix}.episode_median_positive": episodes["median"] is not None
                and _finite(episodes["median"], field="episode median") > 0.0,
                f"{prefix}.positive_episode_concentration_at_most_50pct": (
                    concentration is not None
                    and _finite(concentration, field="episode concentration") <= 0.5
                ),
                f"{prefix}.negative_aapl_year_edge_positive": _finite(
                    summary["negative_aapl_year_edge_sum"],
                    field="negative-year edge",
                )
                > 0.0,
                f"{prefix}.full_account_episode_reconciled": summary[
                    "full_account_episode_reconciled"
                ]
                is True,
            }
        )
        learner_full = _finite(
            summary["full_account_active_log_edge"],
            field=f"confirmation learner full edge[{cost_name}]",
        )
        diffs = {
            name: math.fsum(
                (
                    learner_full,
                    -_finite(
                        fixed[name][cost_name]["policy"]["summary"][
                            "full_account_active_log_edge"
                        ],
                        field=f"confirmation comparator {name}",
                    ),
                )
            )
            for name in FIXED_COMPARATOR_NAMES
        }
        checks[f"{prefix}.full_account_beats_every_fixed_by_gt_0_0001"] = all(
            value > STRICT_INCREMENTAL_EDGE for value in diffs.values()
        )
        comparator_diagnostics[cost_name] = diffs
        clean_year_edges = fixed["exact_union_cash"][cost_name]["xor_summary"][
            "year_edges"
        ]
        checks[f"{prefix}.learner_minus_union_positive_in_at_least_2_years"] = (
            sum(value > 0.0 for value in clean_year_edges.values()) >= 2
        )

    for cost_name in COST_NAMES:
        always_long = fixed["always_long"][cost_name]["policy"]
        try:
            _ledger.assert_always_long_matches_buy_hold(
                always_long["strategy_ledger"], always_long["benchmark_ledger"]
            )
        except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
            raise ContextualExpertAggregationEvaluationError(
                f"confirmation always-LONG control differs at {cost_name}"
            ) from exc

    ablation_diagnostics: dict[str, Any] = {}
    for comparison_name in ABLATION_COMPARISON_NAMES:
        stress = ablations[comparison_name][STRESS_COST_NAME]["xor_summary"]
        concentration = stress["positive_concentration"]
        checks.update(
            {
                f"ablation.{comparison_name}.stress_xor_count_at_least_3": _integer(
                    stress["count"], field="XOR count"
                )
                >= 3,
                f"ablation.{comparison_name}.stress_incremental_edge_gt_0_0001": _finite(
                    stress["sum"], field="XOR aggregate edge"
                )
                > STRICT_INCREMENTAL_EDGE,
                f"ablation.{comparison_name}.stress_positive_years_at_least_2": _integer(
                    stress["positive_year_count"], field="XOR positive years"
                )
                >= 2,
                f"ablation.{comparison_name}.stress_beneficial_rate_at_least_50pct": _finite(
                    stress["positive_rate"], field="XOR beneficial rate"
                )
                >= 0.5,
                f"ablation.{comparison_name}.stress_mean_positive": stress["mean"]
                is not None
                and _finite(stress["mean"], field="XOR mean") > 0.0,
                f"ablation.{comparison_name}.stress_median_positive": stress[
                    "median"
                ]
                is not None
                and _finite(stress["median"], field="XOR median") > 0.0,
                f"ablation.{comparison_name}.stress_concentration_at_most_50pct": (
                    concentration is not None
                    and _finite(concentration, field="XOR concentration") <= 0.5
                ),
            }
        )
        base = ablations[comparison_name][BASE_COST_NAME]["xor_summary"]
        if comparison_name == "online_minus_frozen_2018":
            checks[
                "ablation.online_minus_frozen_2018.base_incremental_edge_gt_0_0001"
            ] = _finite(base["sum"], field="online-frozen base edge") > STRICT_INCREMENTAL_EDGE
        ablation_diagnostics[comparison_name] = {
            BASE_COST_NAME: dict(base),
            STRESS_COST_NAME: dict(stress),
        }

    report = _gate_report(checks)
    report["fixed_comparator_differences"] = comparator_diagnostics
    report["ablation_diagnostics"] = ablation_diagnostics
    report["policy_summaries"] = {
        cost: primary[cost]["summary"] for cost in COST_NAMES
    }
    report["learner_minus_union_year_edges"] = {
        cost: dict(fixed["exact_union_cash"][cost]["xor_summary"]["year_edges"])
        for cost in COST_NAMES
    }
    return report


def require_report_status(report: Mapping[str, Any], *, expected_pass: bool) -> None:
    """Require a top-level result to agree exactly with its gate checks."""

    if not isinstance(report, Mapping):
        raise ContextualExpertAggregationEvaluationError(
            "gate report must be a mapping"
        )
    passed = _strict_bool(report.get("passed"), field="gate report passed")
    expected = _strict_bool(expected_pass, field="expected_pass")
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise ContextualExpertAggregationEvaluationError(
            "gate report checks must be a nonempty mapping"
        )
    recomputed = all(
        _strict_bool(value, field=f"gate check {name}")
        for name, value in checks.items()
    )
    passed_count = _integer(
        report.get("passed_count"), field="gate report passed_count"
    )
    total_count = _integer(
        report.get("total_count"), field="gate report total_count", minimum=1
    )
    failed_checks = report.get("failed_checks")
    expected_failed = [name for name, value in checks.items() if not value]
    if (
        passed_count != sum(bool(value) for value in checks.values())
        or total_count != len(checks)
        or not isinstance(failed_checks, list)
        or failed_checks != expected_failed
    ):
        raise ContextualExpertAggregationEvaluationError(
            "gate report diagnostics disagree with exact checks"
        )
    if passed is not recomputed or passed is not expected:
        raise ContextualExpertAggregationEvaluationError(
            "gate report and top-level status disagree"
        )


__all__ = [
    "ABLATION_COMPARISON_NAMES",
    "BASE_COST_NAME",
    "COMMON_INTEGRITY_CHECKS",
    "CONFIRMATION_ACCOUNT_YEARS",
    "CONFIRMATION_INTEGRITY_CHECKS",
    "CONFIRMATION_YEARS",
    "COST_BPS",
    "COST_NAMES",
    "ContextualExpertAggregationEvaluationError",
    "DEVELOPMENT_FOLDS",
    "DEVELOPMENT_INTEGRITY_CHECKS",
    "DEVELOPMENT_YEARS",
    "EPISODE_COLUMNS",
    "FIXED_COMPARATOR_NAMES",
    "INITIAL_CASH",
    "LEDGER_COLUMNS",
    "MIN_ACTIVE_LOG_EDGE",
    "PAIRWISE_EVIDENCE_KEYS",
    "POLICY_EVIDENCE_KEYS",
    "RECONCILIATION_TOLERANCE",
    "STRESS_COST_NAME",
    "STRICT_INCREMENTAL_EDGE",
    "XOR_COLUMNS",
    "XOR_ORIENTATIONS",
    "apply_confirmation_gates",
    "apply_development_gates",
    "require_report_status",
    "summarize_complete_episodes",
    "summarize_policy_ledgers",
    "summarize_xor_differences",
]
