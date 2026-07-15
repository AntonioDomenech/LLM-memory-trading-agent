"""Pure metrics and gates for the contextual-aggregation continuation audit.

The audit runner supplies already constructed continuous ledgers, complete
cash episodes, signed XOR episodes, and compact online-versus-ablation state
comparisons.  This module performs no file or network I/O.  It delegates the
canonical ledger/episode/XOR replay checks to the frozen generic evaluator and
adds only the preregistered 2019-2023 and continuous 2005-2023 audit metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd

from . import contextual_expert_aggregation as _model
from . import contextual_expert_aggregation_evaluation as _base
from . import contextual_expert_aggregation_ledger as _ledger


BASE_COST_NAME = _base.BASE_COST_NAME
STRESS_COST_NAME = _base.STRESS_COST_NAME
COST_NAMES = _base.COST_NAMES
COST_BPS = _base.COST_BPS

CONTINUOUS_YEARS = tuple(range(2005, 2024))
SUFFIX_YEARS = tuple(range(2019, 2024))
SUFFIX_BLOCKS: Mapping[str, tuple[int, ...]] = {
    "2019_2020": (2019, 2020),
    "2021_2022": (2021, 2022),
    "2023": (2023,),
}

FIXED_COMPARATOR_NAMES = _base.FIXED_COMPARATOR_NAMES
ADAPTIVE_COMPARISON_NAMES = _base.ABLATION_COMPARISON_NAMES
AUDIT_COMPARATOR_NAMES = FIXED_COMPARATOR_NAMES + ADAPTIVE_COMPARISON_NAMES
ONLINE_MINUS_FROZEN = "online_minus_frozen_2018"
EXACT_UNION = "exact_union_cash"
AUDIT_POLICY_NAMES = (
    "online_full",
    "frozen_2018",
    "global_only",
    "lifetime_only",
    *FIXED_COMPARATOR_NAMES,
)
_COMPARISON_BY_POLICY = {
    "frozen_2018": ONLINE_MINUS_FROZEN,
    "global_only": "full_minus_global_only",
    "lifetime_only": "full_minus_lifetime_only",
    **{name: name for name in FIXED_COMPARATOR_NAMES},
}

ADAPTIVE_STATE_COLUMNS = (
    "decision_date",
    "canonical_union_opportunity",
    "online_cash_score",
    "comparator_cash_score",
    "online_action",
    "comparator_action",
)

EDGE_MATERIALITY = 0.001
ZERO_TOLERANCE = 1e-12

ADAPTIVE_STATUSES = frozenset(
    {
        "unexercised",
        "exercised_insufficient_evidence",
        "exercised_positive",
        "exercised_negative",
        "exercised_flat",
    }
)


class ContextualExpertAggregationAuditEvaluationError(
    _base.ContextualExpertAggregationEvaluationError
):
    """Raised when audit-only scoring evidence is malformed."""


def _finite(value: Any, *, field: str) -> float:
    try:
        return _base._finite(value, field=field)
    except _base.ContextualExpertAggregationEvaluationError as exc:
        raise ContextualExpertAggregationAuditEvaluationError(str(exc)) from exc


def _exact_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} must be an exact integer >= {minimum}"
        )
    return value


def _exact_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} must be an exact boolean"
        )
    return value


def _exact_cost_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(COST_NAMES):
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} must contain exactly {list(COST_NAMES)}"
        )
    return value


def _criterion_report(checks: Mapping[str, bool | None]) -> dict[str, Any]:
    """Build a gate report while keeping explicit not-applicable diagnostics."""

    if not isinstance(checks, Mapping) or not checks:
        raise ContextualExpertAggregationAuditEvaluationError(
            "criterion report requires nonempty checks"
        )
    normalized: dict[str, bool | None] = {}
    for name, value in sorted(checks.items()):
        if not isinstance(name, str) or not name:
            raise ContextualExpertAggregationAuditEvaluationError(
                "criterion check names must be nonempty strings"
            )
        if value is not None:
            _exact_bool(value, field=f"criterion.{name}")
        normalized[name] = value
    applicable = {name: value for name, value in normalized.items() if value is not None}
    failed = [name for name, value in applicable.items() if value is False]
    not_applicable = [name for name, value in normalized.items() if value is None]
    return {
        "passed": not failed and bool(applicable),
        "passed_count": sum(value is True for value in applicable.values()),
        "applicable_count": len(applicable),
        "total_count": len(normalized),
        "checks": normalized,
        "failed_checks": failed,
        "not_applicable_checks": not_applicable,
    }


def _top_k_removed(values: Sequence[float], *, k: int) -> float | None:
    count = _exact_int(k, field="top-k count", minimum=1)
    clean = [_finite(value, field="episode edge") for value in values]
    if len(clean) < count:
        return None
    largest = sorted(clean, reverse=True)[:count]
    return float(math.fsum((math.fsum(clean), -math.fsum(largest))))


def _maximum_drawdown(ledger: pd.DataFrame, *, field: str) -> float:
    if not isinstance(ledger, pd.DataFrame) or ledger.empty:
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} must be a nonempty canonical ledger"
        )
    values = [_finite(value, field=f"{field}.drawdown") for value in ledger["drawdown"]]
    result = min(values)
    if result > 0.0:
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} drawdown must use nonpositive values"
        )
    return float(result)


def _suffix_block_edges(summary: Mapping[str, Any]) -> dict[str, float]:
    year_edges = summary["episodes"]["year_edges"]
    result = {
        name: float(math.fsum(_finite(year_edges[str(year)], field="year edge") for year in years))
        for name, years in SUFFIX_BLOCKS.items()
    }
    if not math.isclose(
        math.fsum(result.values()),
        _finite(summary["reporting_active_log_edge"], field="suffix reporting edge"),
        rel_tol=0.0,
        abs_tol=ZERO_TOLERANCE,
    ):
        raise ContextualExpertAggregationAuditEvaluationError(
            "suffix block edges do not reconcile to the entry-attributed suffix edge"
        )
    return result


def _policy_cost_report(
    evaluated: Mapping[str, Any], *, cost_name: str
) -> dict[str, Any]:
    cost_bps = COST_BPS[cost_name]
    suffix = evaluated["summary"]
    continuous = _base.summarize_policy_ledgers(
        evaluated["strategy_ledger"],
        evaluated["benchmark_ledger"],
        evaluated["episodes"],
        reporting_years=CONTINUOUS_YEARS,
        account_years=CONTINUOUS_YEARS,
        cost_bps=cost_bps,
        folds=None,
    )
    _base._require_summary_period(
        _base._validated_policy_summary(
            continuous, field=f"audit.{cost_name}.continuous"
        ),
        years=CONTINUOUS_YEARS,
        fold_names=set(),
        field=f"audit.{cost_name}.continuous",
    )

    suffix_episodes = suffix["episodes"]
    suffix_concentration = suffix_episodes["positive_concentration"]
    suffix_blocks = _suffix_block_edges(suffix)
    suffix_negative_years = list(suffix["negative_aapl_years"])
    suffix_negative_status = (
        "observed_positive"
        if suffix_negative_years and suffix["negative_aapl_year_edge_sum"] > 0.0
        else "observed_nonpositive"
        if suffix_negative_years
        else "not_applicable_no_negative_aapl_years"
    )
    suffix_checks: dict[str, bool | None] = {
        "entry_attributed_active_log_edge_gt_0_001": (
            _finite(suffix["reporting_active_log_edge"], field="suffix edge")
            > EDGE_MATERIALITY
        ),
        "positive_years_at_least_3_of_5": (
            _exact_int(suffix["positive_year_count"], field="suffix positive years")
            >= 3
        ),
        "edge_after_removing_best_year_positive": (
            _finite(suffix["edge_after_removing_best_year"], field="suffix edge after best year")
            > 0.0
        ),
        "positive_blocks_at_least_2_of_3": sum(value > 0.0 for value in suffix_blocks.values())
        >= 2,
        "complete_episodes_at_least_5": (
            _exact_int(suffix_episodes["count"], field="suffix episode count") >= 5
        ),
        "beneficial_episode_rate_at_least_50pct": (
            _finite(suffix_episodes["positive_rate"], field="suffix episode rate") >= 0.50
        ),
        "episode_mean_positive": suffix_episodes["mean"] is not None
        and _finite(suffix_episodes["mean"], field="suffix episode mean") > 0.0,
        "episode_median_positive": suffix_episodes["median"] is not None
        and _finite(suffix_episodes["median"], field="suffix episode median") > 0.0,
        "positive_episode_concentration_at_most_50pct": (
            suffix_concentration is not None
            and _finite(suffix_concentration, field="suffix concentration") <= 0.50
        ),
        "negative_aapl_year_edge_positive_if_applicable": (
            _finite(suffix["negative_aapl_year_edge_sum"], field="suffix negative-year edge")
            > 0.0
        )
        if suffix_negative_years
        else None,
    }
    suffix_criterion = _criterion_report(suffix_checks)

    continuous_episodes = continuous["episodes"]
    all_episode_values = [
        _finite(value, field="continuous episode edge")
        for value in evaluated["episodes"]["net_active_log_edge"].tolist()
    ]
    after_top_five = _top_k_removed(all_episode_values, k=5)
    strategy_drawdown = _maximum_drawdown(
        evaluated["strategy_ledger"], field="strategy ledger"
    )
    benchmark_drawdown = _maximum_drawdown(
        evaluated["benchmark_ledger"], field="benchmark ledger"
    )
    continuous_concentration = continuous_episodes["positive_concentration"]
    continuous_checks: dict[str, bool | None] = {
        "entry_attributed_active_log_edge_gt_0_001": (
            _finite(continuous["reporting_active_log_edge"], field="continuous edge")
            > EDGE_MATERIALITY
        ),
        "positive_years_at_least_11_of_19": (
            _exact_int(
                continuous["positive_year_count"], field="continuous positive years"
            )
            >= 11
        ),
        "edge_after_removing_best_year_positive": (
            _finite(
                continuous["edge_after_removing_best_year"],
                field="continuous edge after best year",
            )
            > 0.0
        ),
        "negative_aapl_years_present_and_edge_positive": bool(
            continuous["negative_aapl_years"]
        )
        and _finite(
            continuous["negative_aapl_year_edge_sum"],
            field="continuous negative-year edge",
        )
        > 0.0,
        "complete_episodes_at_least_125": (
            _exact_int(continuous_episodes["count"], field="continuous episode count")
            >= 125
        ),
        "beneficial_episode_rate_at_least_55pct": (
            _finite(
                continuous_episodes["positive_rate"],
                field="continuous episode rate",
            )
            >= 0.55
        ),
        "episode_mean_positive": continuous_episodes["mean"] is not None
        and _finite(
            continuous_episodes["mean"], field="continuous episode mean"
        )
        > 0.0,
        "episode_median_positive": continuous_episodes["median"] is not None
        and _finite(
            continuous_episodes["median"], field="continuous episode median"
        )
        > 0.0,
        "edge_after_removing_five_largest_episodes_positive": (
            after_top_five is not None and after_top_five > 0.0
        ),
        "positive_episode_concentration_at_most_25pct": (
            continuous_concentration is not None
            and _finite(
                continuous_concentration, field="continuous concentration"
            )
            <= 0.25
        ),
        "strategy_max_drawdown_no_worse_than_aapl": strategy_drawdown
        >= benchmark_drawdown,
    }
    continuous_criterion = _criterion_report(continuous_checks)

    return {
        "cost_name": cost_name,
        "cost_bps": cost_bps,
        "suffix_2019_2023": {
            "entry_attributed_active_log_edge": suffix["reporting_active_log_edge"],
            "ledger_boundary_active_log_edge": suffix[
                "reporting_ledger_boundary_active_log_edge"
            ],
            "annual": suffix["annual"],
            "block_edges": suffix_blocks,
            "positive_block_count": sum(value > 0.0 for value in suffix_blocks.values()),
            "edge_after_removing_best_year": suffix["edge_after_removing_best_year"],
            "negative_aapl_years": suffix_negative_years,
            "negative_aapl_year_edge_sum": suffix["negative_aapl_year_edge_sum"],
            "negative_aapl_year_support_status": suffix_negative_status,
            "episodes": suffix_episodes,
            "criterion": suffix_criterion,
        },
        "continuous_2005_2023": {
            "entry_attributed_active_log_edge": continuous["reporting_active_log_edge"],
            "ledger_boundary_active_log_edge": continuous[
                "reporting_ledger_boundary_active_log_edge"
            ],
            "annual": continuous["annual"],
            "edge_after_removing_best_year": continuous[
                "edge_after_removing_best_year"
            ],
            "edge_after_removing_five_largest_episodes": after_top_five,
            "negative_aapl_years": continuous["negative_aapl_years"],
            "negative_aapl_year_edge_sum": continuous[
                "negative_aapl_year_edge_sum"
            ],
            "strategy_max_drawdown": strategy_drawdown,
            "aapl_max_drawdown": benchmark_drawdown,
            "episodes": continuous_episodes,
            "criterion": continuous_criterion,
        },
        "post_rejection_2019_2023_pass": suffix_criterion["passed"],
        "continuous_2005_2023_robustness_pass": continuous_criterion["passed"],
    }


def _all_policy_diagnostics(
    primary: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Report every required year/block for every arm and fixed policy."""

    by_cost: dict[str, Any] = {}
    for cost_name in COST_NAMES:
        policies: dict[str, Any] = {}
        for policy_name in AUDIT_POLICY_NAMES:
            evaluated = (
                primary[cost_name]
                if policy_name == "online_full"
                else comparisons[_COMPARISON_BY_POLICY[policy_name]][cost_name][
                    "policy"
                ]
            )
            observed_names = set(
                evaluated["strategy_ledger"]["policy_name"].astype(str).tolist()
            )
            if observed_names != {policy_name}:
                raise ContextualExpertAggregationAuditEvaluationError(
                    f"audit policy diagnostic identity changed for {policy_name}"
                )
            suffix = _base._validated_policy_summary(
                evaluated["summary"],
                field=f"audit.{cost_name}.{policy_name}.suffix",
            )
            _base._require_summary_period(
                suffix,
                years=SUFFIX_YEARS,
                fold_names=set(),
                field=f"audit.{cost_name}.{policy_name}.suffix",
            )
            continuous = _base.summarize_policy_ledgers(
                evaluated["strategy_ledger"],
                evaluated["benchmark_ledger"],
                evaluated["episodes"],
                reporting_years=CONTINUOUS_YEARS,
                account_years=CONTINUOUS_YEARS,
                cost_bps=COST_BPS[cost_name],
                folds=None,
            )
            _base._require_summary_period(
                _base._validated_policy_summary(
                    continuous,
                    field=f"audit.{cost_name}.{policy_name}.continuous",
                ),
                years=CONTINUOUS_YEARS,
                fold_names=set(),
                field=f"audit.{cost_name}.{policy_name}.continuous",
            )
            policies[policy_name] = {
                "policy_name": policy_name,
                "suffix_2019_2023": {
                    **dict(suffix),
                    "fixed_block_edges": _suffix_block_edges(suffix),
                },
                "continuous_2005_2023": continuous,
            }
        by_cost[cost_name] = {
            "cost_bps": COST_BPS[cost_name],
            "policy_order": list(AUDIT_POLICY_NAMES),
            "policies": policies,
        }
    return {
        "policy_order": list(AUDIT_POLICY_NAMES),
        "by_cost": by_cost,
    }


def _canonical_state_frame(
    value: Any,
    *,
    primary_ledger: pd.DataFrame,
    comparator_ledger: pd.DataFrame,
    field: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(value, pd.DataFrame) or tuple(value.columns) != ADAPTIVE_STATE_COLUMNS:
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} must have the exact adaptive state schema"
        )
    frame = value.copy()
    if frame.empty:
        raise ContextualExpertAggregationAuditEvaluationError(f"{field} is empty")
    parsed_dates: list[pd.Timestamp] = []
    for position, raw in enumerate(frame["decision_date"].tolist()):
        if type(raw) is not str:
            raise ContextualExpertAggregationAuditEvaluationError(
                f"{field}[{position}].decision_date must be a canonical date string"
            )
        try:
            parsed = date.fromisoformat(raw)
        except ValueError as exc:
            raise ContextualExpertAggregationAuditEvaluationError(
                f"{field}[{position}].decision_date is invalid"
            ) from exc
        if parsed.isoformat() != raw:
            raise ContextualExpertAggregationAuditEvaluationError(
                f"{field}[{position}].decision_date is not canonical"
            )
        parsed_dates.append(pd.Timestamp(parsed))
    dates = pd.Series(parsed_dates, index=frame.index, dtype="datetime64[ns]")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} dates must be unique and increasing"
        )
    if tuple(sorted(set(dates.dt.year.astype(int)))) != CONTINUOUS_YEARS:
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} must cover every audit account year"
        )

    for position, row in enumerate(frame.to_dict(orient="records")):
        opportunity = _exact_bool(
            row["canonical_union_opportunity"],
            field=f"{field}[{position}].canonical_union_opportunity",
        )
        for role in ("online", "comparator"):
            score = _finite(row[f"{role}_cash_score"], field=f"{field}[{position}].{role}_cash_score")
            if not 0.0 <= score <= 1.0 or type(row[f"{role}_cash_score"]) is not float:
                raise ContextualExpertAggregationAuditEvaluationError(
                    f"{field}[{position}].{role}_cash_score must be an exact float in [0,1]"
                )
            action = row[f"{role}_action"]
            if action not in {"LONG", "CASH"}:
                raise ContextualExpertAggregationAuditEvaluationError(
                    f"{field}[{position}].{role}_action is invalid"
                )
            expected = _model.resolve_cash_action(score, opportunity=opportunity)
            if action != expected:
                raise ContextualExpertAggregationAuditEvaluationError(
                    f"{field}[{position}].{role}_action does not match the frozen threshold"
                )

    # Forecast decisions are made on each completed session and are queued for
    # the next fill.  The ledger's ``decision_date`` therefore describes the
    # target arriving *at* the current fill (and is empty at inception), while
    # ``close_decision_target_exposure`` is the forecast made on this row.
    # Align the compact forecast evidence to the row's session/fill date and
    # close decision, including the final queued decision.
    ledger_dates = primary_ledger["fill_date"].tolist()
    comparator_dates = comparator_ledger["fill_date"].tolist()
    if frame["decision_date"].tolist() != ledger_dates or ledger_dates != comparator_dates:
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} does not align exactly with both ledger decision streams"
        )
    action_target = {"CASH": 0, "LONG": 1}
    online_targets = [action_target[value] for value in frame["online_action"]]
    comparator_targets = [action_target[value] for value in frame["comparator_action"]]
    if online_targets != primary_ledger["close_decision_target_exposure"].tolist():
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} online actions differ from the online ledger"
        )
    if comparator_targets != comparator_ledger[
        "close_decision_target_exposure"
    ].tolist():
        raise ContextualExpertAggregationAuditEvaluationError(
            f"{field} comparator actions differ from the comparator ledger"
        )
    return frame, dates


def _state_period_summary(
    frame: pd.DataFrame, dates: pd.Series, *, years: Sequence[int]
) -> dict[str, Any]:
    selected = frame.loc[dates.dt.year.isin(tuple(years))]
    selected_dates = dates.loc[selected.index]
    score_difference = selected["online_cash_score"] != selected["comparator_cash_score"]
    action_xor = selected["online_action"] != selected["comparator_action"]
    threshold = action_xor.copy()
    crossing_dates = selected.loc[threshold, "decision_date"].tolist()
    crossing_years = sorted(set(selected_dates.loc[threshold].dt.year.astype(int).tolist()))
    return {
        "forecast_row_count": len(selected),
        "cash_score_difference_count": int(score_difference.sum()),
        "threshold_crossing_count": int(threshold.sum()),
        "action_xor_decision_count": int(action_xor.sum()),
        "first_threshold_crossing_date": crossing_dates[0] if crossing_dates else None,
        "last_threshold_crossing_date": crossing_dates[-1] if crossing_dates else None,
        "distinct_threshold_crossing_years": crossing_years,
    }


def classify_adaptive_status(
    *,
    cash_score_difference_count: int,
    threshold_crossing_count: int,
    xor_episode_count: int,
    distinct_xor_entry_year_count: int,
    stress_incremental_edge: float,
) -> str:
    """Apply the preregistered 10-bps suffix adaptive classification."""

    score_differences = _exact_int(
        cash_score_difference_count, field="cash score difference count"
    )
    crossings = _exact_int(
        threshold_crossing_count, field="threshold crossing count"
    )
    episodes = _exact_int(xor_episode_count, field="XOR episode count")
    years = _exact_int(
        distinct_xor_entry_year_count, field="distinct XOR entry year count"
    )
    edge = _finite(stress_incremental_edge, field="stress incremental edge")
    if score_differences == 0 or crossings == 0 or episodes == 0:
        return "unexercised"
    if episodes < 5 or years < 2:
        return "exercised_insufficient_evidence"
    if edge > ZERO_TOLERANCE:
        return "exercised_positive"
    if edge < -ZERO_TOLERANCE:
        return "exercised_negative"
    return "exercised_flat"


def _xor_distinct_entry_years(frame: pd.DataFrame, *, years: Sequence[int]) -> list[int]:
    if frame.empty:
        return []
    dates = pd.to_datetime(frame["entry_fill_date"], format="%Y-%m-%d", exact=True)
    return sorted(set(dates.loc[dates.dt.year.isin(tuple(years))].dt.year.astype(int)))


def _adaptive_report(
    comparison_name: str,
    *,
    comparison_by_cost: Mapping[str, Mapping[str, Any]],
    state_frame: pd.DataFrame,
    primary_by_cost: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    stress_pair = comparison_by_cost[STRESS_COST_NAME]
    canonical_state, state_dates = _canonical_state_frame(
        state_frame,
        primary_ledger=primary_by_cost[STRESS_COST_NAME]["strategy_ledger"],
        comparator_ledger=stress_pair["policy"]["strategy_ledger"],
        field=f"adaptive state {comparison_name}",
    )
    suffix_state = _state_period_summary(
        canonical_state, state_dates, years=SUFFIX_YEARS
    )
    continuous_state = _state_period_summary(
        canonical_state, state_dates, years=CONTINUOUS_YEARS
    )
    if (
        suffix_state["threshold_crossing_count"]
        != suffix_state["action_xor_decision_count"]
        or continuous_state["threshold_crossing_count"]
        != continuous_state["action_xor_decision_count"]
    ):
        raise ContextualExpertAggregationAuditEvaluationError(
            f"adaptive state {comparison_name} threshold and action XOR counts differ"
        )

    by_cost: dict[str, Any] = {}
    for cost_name in COST_NAMES:
        pair = comparison_by_cost[cost_name]
        suffix_years = _xor_distinct_entry_years(pair["xor"], years=SUFFIX_YEARS)
        continuous_years = _xor_distinct_entry_years(
            pair["xor"], years=CONTINUOUS_YEARS
        )
        by_cost[cost_name] = {
            "suffix_2019_2023": {
                "incremental_active_log_edge": pair["xor_summary"]["sum"],
                "complete_xor_episode_count": pair["xor_summary"]["count"],
                "distinct_xor_entry_years": suffix_years,
                "xor_summary": pair["xor_summary"],
            },
            "continuous_2005_2023": {
                "incremental_active_log_edge": pair["full_xor_summary"]["sum"],
                "complete_xor_episode_count": pair["full_xor_summary"]["count"],
                "distinct_xor_entry_years": continuous_years,
                "xor_summary": pair["full_xor_summary"],
            },
        }

    stress_suffix = by_cost[STRESS_COST_NAME]["suffix_2019_2023"]
    status = classify_adaptive_status(
        cash_score_difference_count=suffix_state["cash_score_difference_count"],
        threshold_crossing_count=suffix_state["threshold_crossing_count"],
        xor_episode_count=stress_suffix["complete_xor_episode_count"],
        distinct_xor_entry_year_count=len(stress_suffix["distinct_xor_entry_years"]),
        stress_incremental_edge=stress_suffix["incremental_active_log_edge"],
    )
    return {
        "comparison_name": comparison_name,
        "classification_basis": "stress_10bps_suffix_2019_2023",
        "adaptive_status": status,
        "useful_continual_learning_demonstrated": status == "exercised_positive",
        "state_differences": {
            "suffix_2019_2023": suffix_state,
            "continuous_2005_2023": continuous_state,
        },
        "by_cost": by_cost,
    }


def aggregate_decision_statuses(
    *,
    post_rejection_by_cost: Mapping[str, bool],
    continuous_by_cost: Mapping[str, bool],
    online_minus_frozen_suffix_edges: Mapping[str, float],
    online_minus_union_suffix_edges: Mapping[str, float],
    online_minus_frozen_adaptive_status: str,
) -> dict[str, Any]:
    """Aggregate per-cost criteria without allowing either cost to disappear."""

    post = _exact_cost_mapping(post_rejection_by_cost, field="post-rejection statuses")
    continuous = _exact_cost_mapping(continuous_by_cost, field="continuous statuses")
    frozen = _exact_cost_mapping(
        online_minus_frozen_suffix_edges, field="online-minus-frozen suffix edges"
    )
    union = _exact_cost_mapping(
        online_minus_union_suffix_edges, field="online-minus-union suffix edges"
    )
    post_flags = {
        cost: _exact_bool(post[cost], field=f"post-rejection status[{cost}]")
        for cost in COST_NAMES
    }
    continuous_flags = {
        cost: _exact_bool(continuous[cost], field=f"continuous status[{cost}]")
        for cost in COST_NAMES
    }
    frozen_edges = {
        cost: _finite(frozen[cost], field=f"online-minus-frozen edge[{cost}]")
        for cost in COST_NAMES
    }
    union_edges = {
        cost: _finite(union[cost], field=f"online-minus-union edge[{cost}]")
        for cost in COST_NAMES
    }
    if online_minus_frozen_adaptive_status not in ADAPTIVE_STATUSES:
        raise ContextualExpertAggregationAuditEvaluationError(
            "online-minus-frozen adaptive status is invalid"
        )
    post_pass = all(post_flags.values())
    continuous_pass = all(continuous_flags.values())
    historical = post_pass and continuous_pass
    learning = (
        historical
        and online_minus_frozen_adaptive_status == "exercised_positive"
        and all(value > 0.0 for value in frozen_edges.values())
        and all(value > 0.0 for value in union_edges.values())
    )
    return {
        "post_rejection_2019_2023_by_cost": post_flags,
        "continuous_2005_2023_robustness_by_cost": continuous_flags,
        "post_rejection_2019_2023_pass": post_pass,
        "continuous_2005_2023_robustness_pass": continuous_pass,
        "historical_policy_candidate_for_2024_audit": historical,
        "learning_candidate_for_2024_audit": learning,
        "online_minus_frozen_suffix_edges": frozen_edges,
        "online_minus_union_suffix_edges": union_edges,
        "online_minus_frozen_adaptive_status": online_minus_frozen_adaptive_status,
    }


def apply_audit_gates(
    policy_evidence: Mapping[str, Mapping[str, Any]],
    *,
    comparator_evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
    adaptive_state_differences: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Replay all canonical evidence and apply the preregistered audit gates."""

    primary = _base._evaluate_primary_by_cost(
        policy_evidence, stage="confirmation"
    )
    comparisons = _base._evaluate_pairwise_inventory(
        comparator_evidence,
        names=AUDIT_COMPARATOR_NAMES,
        primary_by_cost=primary,
        stage="confirmation",
        field="post-rejection audit comparators",
    )
    if not isinstance(adaptive_state_differences, Mapping) or set(
        adaptive_state_differences
    ) != set(ADAPTIVE_COMPARISON_NAMES):
        raise ContextualExpertAggregationAuditEvaluationError(
            "adaptive state evidence does not have the exact comparison inventory"
        )

    for comparison_name in ADAPTIVE_COMPARISON_NAMES:
        for cost_name in COST_NAMES:
            _base._require_shared_account_prefix(
                primary[cost_name]["strategy_ledger"],
                comparisons[comparison_name][cost_name]["policy"]["strategy_ledger"],
                cutoff_fill_date="2018-12-31",
                field=f"audit ablation {comparison_name}.{cost_name}",
            )

    for cost_name in COST_NAMES:
        always_long = comparisons["always_long"][cost_name]["policy"]
        try:
            _ledger.assert_always_long_matches_buy_hold(
                always_long["strategy_ledger"], always_long["benchmark_ledger"]
            )
        except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
            raise ContextualExpertAggregationAuditEvaluationError(
                f"audit always-LONG control differs at {cost_name}"
            ) from exc

    policy_reports = {
        cost: _policy_cost_report(primary[cost], cost_name=cost) for cost in COST_NAMES
    }
    all_policy_diagnostics = _all_policy_diagnostics(primary, comparisons)

    comparator_diagnostics: dict[str, Any] = {}
    for name in FIXED_COMPARATOR_NAMES:
        comparator_diagnostics[name] = {
            cost: {
                "suffix_2019_2023_comparator_active_log_edge": comparisons[name][cost][
                    "policy"
                ]["summary"]["reporting_active_log_edge"],
                "suffix_2019_2023_online_minus_comparator_edge": comparisons[name][
                    cost
                ]["xor_summary"]["sum"],
                "continuous_2005_2023_comparator_active_log_edge": comparisons[name][
                    cost
                ]["policy"]["summary"]["full_account_active_log_edge"],
                "continuous_2005_2023_online_minus_comparator_edge": comparisons[name][
                    cost
                ]["full_incremental_edge"],
            }
            for cost in COST_NAMES
        }

    best_fixed: dict[str, Any] = {}
    for cost in COST_NAMES:
        suffix_best = max(
            FIXED_COMPARATOR_NAMES,
            key=lambda name: comparator_diagnostics[name][cost][
                "suffix_2019_2023_comparator_active_log_edge"
            ],
        )
        continuous_best = max(
            FIXED_COMPARATOR_NAMES,
            key=lambda name: comparator_diagnostics[name][cost][
                "continuous_2005_2023_comparator_active_log_edge"
            ],
        )
        best_fixed[cost] = {
            "suffix_2019_2023": {
                "name": suffix_best,
                **comparator_diagnostics[suffix_best][cost],
            },
            "continuous_2005_2023": {
                "name": continuous_best,
                **comparator_diagnostics[continuous_best][cost],
            },
        }

    adaptive = {
        name: _adaptive_report(
            name,
            comparison_by_cost=comparisons[name],
            state_frame=adaptive_state_differences[name],
            primary_by_cost=primary,
        )
        for name in ADAPTIVE_COMPARISON_NAMES
    }

    frozen_edges = {
        cost: adaptive[ONLINE_MINUS_FROZEN]["by_cost"][cost]["suffix_2019_2023"][
            "incremental_active_log_edge"
        ]
        for cost in COST_NAMES
    }
    union_edges = {
        cost: comparisons[EXACT_UNION][cost]["xor_summary"]["sum"]
        for cost in COST_NAMES
    }
    statuses = aggregate_decision_statuses(
        post_rejection_by_cost={
            cost: policy_reports[cost]["post_rejection_2019_2023_pass"]
            for cost in COST_NAMES
        },
        continuous_by_cost={
            cost: policy_reports[cost]["continuous_2005_2023_robustness_pass"]
            for cost in COST_NAMES
        },
        online_minus_frozen_suffix_edges=frozen_edges,
        online_minus_union_suffix_edges=union_edges,
        online_minus_frozen_adaptive_status=adaptive[ONLINE_MINUS_FROZEN][
            "adaptive_status"
        ],
    )
    return {
        "policy_by_cost": policy_reports,
        "all_policy_diagnostics": all_policy_diagnostics,
        "fixed_comparator_diagnostics": comparator_diagnostics,
        "best_fixed_comparator": best_fixed,
        "adaptive_value": adaptive,
        "decision_statuses": statuses,
    }


__all__ = [
    "ADAPTIVE_COMPARISON_NAMES",
    "ADAPTIVE_STATE_COLUMNS",
    "ADAPTIVE_STATUSES",
    "AUDIT_COMPARATOR_NAMES",
    "AUDIT_POLICY_NAMES",
    "BASE_COST_NAME",
    "CONTINUOUS_YEARS",
    "COST_BPS",
    "COST_NAMES",
    "ContextualExpertAggregationAuditEvaluationError",
    "EDGE_MATERIALITY",
    "FIXED_COMPARATOR_NAMES",
    "STRESS_COST_NAME",
    "SUFFIX_BLOCKS",
    "SUFFIX_YEARS",
    "ZERO_TOLERANCE",
    "aggregate_decision_statuses",
    "apply_audit_gates",
    "classify_adaptive_status",
]
