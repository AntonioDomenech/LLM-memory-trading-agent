"""Pure economic evaluation for the frozen-policy calendar-2024 audit.

The caller supplies already-built continuous ledgers and their deterministically
extracted CASH/XOR evidence.  This module performs no file or network I/O and
does not import or invoke the policy model.  It validates the supplied economic
primitives against the inherited ledger implementation, computes unrounded
same-boundary metrics, and applies only the preregistered gates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from . import contextual_expert_aggregation_evaluation as _base
from . import contextual_expert_aggregation_ledger as _ledger
from . import contextual_expert_aggregation_2024_audit_evidence as _evidence


CONTRACT_VERSION = "aapl-causal-contextual-expert-aggregation-2024-audit-v2"
EVIDENCE_CLASSIFICATION = "post_hoc_frozen_policy_2024_replication"

BASE_COST_NAME = _base.BASE_COST_NAME
STRESS_COST_NAME = _base.STRESS_COST_NAME
COST_NAMES = _base.COST_NAMES
COST_BPS = _base.COST_BPS

AUDIT_YEAR = 2024
ACCOUNT_YEARS = tuple(range(2005, AUDIT_YEAR + 1))
QUARTERS: Mapping[str, tuple[int, ...]] = {
    "Q1": (1, 2, 3),
    "Q2": (4, 5, 6),
    "Q3": (7, 8, 9),
    "Q4": (10, 11, 12),
}

LEAD_POLICY = "frozen_2023_lead"
SHADOW_POLICY = "online_2024_shadow"
AAPL_POLICY = "aapl_buy_hold"
ALWAYS_LONG_POLICY = "always_long"
EXACT_UNION_POLICY = "exact_union_cash"
CONTEXTUAL_POLICY = "contextual_only"
POLICY_NAMES = (
    LEAD_POLICY,
    SHADOW_POLICY,
    AAPL_POLICY,
    ALWAYS_LONG_POLICY,
    EXACT_UNION_POLICY,
    CONTEXTUAL_POLICY,
)
SIMPLE_COMPARATOR_NAMES = (CONTEXTUAL_POLICY, EXACT_UNION_POLICY)

# Both continuation scenarios retain the inherited internal policy identity.
_INTERNAL_POLICY_NAMES: Mapping[str, str] = {
    LEAD_POLICY: "online_full",
    SHADOW_POLICY: "online_full",
    AAPL_POLICY: AAPL_POLICY,
    ALWAYS_LONG_POLICY: ALWAYS_LONG_POLICY,
    EXACT_UNION_POLICY: EXACT_UNION_POLICY,
    CONTEXTUAL_POLICY: CONTEXTUAL_POLICY,
}

POLICY_EVIDENCE_KEYS = frozenset({"strategy_ledger", "cash_episodes"})

EDGE_MATERIALITY = 0.001
RECONCILIATION_TOLERANCE = 1e-10
ZERO_TOLERANCE = 1e-12

PASS_STATUS = "POST_HOC_FROZEN_POLICY_2024_REPLICATION_PASS"
AAPL_ONLY_STATUS = "AAPL_BEATEN_BUT_MODEL_NOT_SELECTED"
INTEGRITY_REJECTION_STATUS = "REJECTED_2024_INTEGRITY"
REJECTION_STATUS = "REJECTED_2024"

LEARNING_STATUSES = frozenset(
    {
        "unexercised",
        "exercised_insufficient_evidence",
        "exercised_positive",
        "exercised_negative",
        "exercised_flat",
    }
)

_GENERIC_TO_CONTRACT_ORIENTATION = {
    _ledger.PRIMARY_CASH_COMPARATOR_LONG: "shadow_cash_lead_long",
    _ledger.PRIMARY_LONG_COMPARATOR_CASH: "shadow_long_lead_cash",
    "equal": "equal",
}


class ContextualExpertAggregation2024AuditEvaluationError(
    _base.ContextualExpertAggregationEvaluationError
):
    """Raised when 2024 audit economic evidence violates the contract."""


def _finite(value: Any, *, field: str) -> float:
    try:
        return _base._finite(value, field=field)
    except _base.ContextualExpertAggregationEvaluationError as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(str(exc)) from exc


def _exact_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} must be an exact integer >= {minimum}"
        )
    return value


def _exact_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} must be an exact boolean"
        )
    return value


def _strict_keys(value: Any, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} must contain exactly {sorted(expected)}"
        )
    return value


def _cost_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    return _strict_keys(value, set(COST_NAMES), field=field)


def _quarter_mapping(value: Any, *, field: str) -> dict[str, float]:
    raw = _strict_keys(value, set(QUARTERS), field=field)
    return {
        quarter: _finite(raw[quarter], field=f"{field}.{quarter}")
        for quarter in QUARTERS
    }


def _criterion_report(checks: Mapping[str, bool | None]) -> dict[str, Any]:
    if not isinstance(checks, Mapping) or not checks:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            "criterion checks must be a nonempty mapping"
        )
    normalized: dict[str, bool | None] = {}
    for name, value in checks.items():
        if not isinstance(name, str) or not name:
            raise ContextualExpertAggregation2024AuditEvaluationError(
                "criterion names must be nonempty strings"
            )
        if value is not None:
            _exact_bool(value, field=f"criterion.{name}")
        normalized[name] = value
    applicable = {name: value for name, value in normalized.items() if value is not None}
    failed = [name for name, value in applicable.items() if value is False]
    return {
        "passed": bool(applicable) and not failed,
        "passed_count": sum(value is True for value in applicable.values()),
        "applicable_count": len(applicable),
        "total_count": len(normalized),
        "checks": normalized,
        "failed_checks": failed,
        "not_applicable_checks": [
            name for name, value in normalized.items() if value is None
        ],
    }


def evaluate_aapl_policy_gate(
    *,
    full_year_active_log_edge: float,
    quarter_active_log_edges: Mapping[str, float],
    aapl_quarter_log_returns: Mapping[str, float],
) -> dict[str, Any]:
    """Apply one cost arm's exact frozen-lead-versus-AAPL gate."""

    full_edge = _finite(full_year_active_log_edge, field="full-year active edge")
    quarter_edges = _quarter_mapping(
        quarter_active_log_edges, field="quarter active edges"
    )
    benchmark_returns = _quarter_mapping(
        aapl_quarter_log_returns, field="AAPL quarter log returns"
    )
    if abs(math.fsum(quarter_edges.values()) - full_edge) > RECONCILIATION_TOLERANCE:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            "quarter active edges do not reconcile to the full-year active edge"
        )
    positive_quarters = [
        quarter for quarter, value in quarter_edges.items() if value > 0.0
    ]
    edge_after_best = float(
        math.fsum((full_edge, -max(quarter_edges.values())))
    )
    negative_quarters = [
        quarter
        for quarter, value in benchmark_returns.items()
        if value < -ZERO_TOLERANCE
    ]
    negative_edge = float(math.fsum(quarter_edges[name] for name in negative_quarters))
    checks: dict[str, bool | None] = {
        "full_year_active_log_edge_gt_0_001": full_edge > EDGE_MATERIALITY,
        "positive_quarters_at_least_2_of_4": len(positive_quarters) >= 2,
        "edge_after_removing_best_quarter_positive": edge_after_best > 0.0,
        "negative_aapl_quarter_edge_positive_if_applicable": (
            negative_edge > 0.0 if negative_quarters else None
        ),
    }
    criterion = _criterion_report(checks)
    return {
        **criterion,
        "full_year_active_log_edge": full_edge,
        "quarter_active_log_edges": quarter_edges,
        "positive_quarters": positive_quarters,
        "positive_quarter_count": len(positive_quarters),
        "edge_after_removing_best_quarter": edge_after_best,
        "aapl_quarter_log_returns": benchmark_returns,
        "negative_aapl_quarters": negative_quarters,
        "negative_aapl_quarter_edge_sum": negative_edge,
        "negative_aapl_quarter_support_status": (
            "observed_positive"
            if negative_quarters and negative_edge > 0.0
            else "observed_nonpositive"
            if negative_quarters
            else "not_applicable_no_negative_aapl_quarters"
        ),
    }


def evaluate_simple_policy_superiority(
    *,
    lead_full_year_log_return: float,
    comparator_full_year_log_returns: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the stress-cost lead-versus-simple-policy gate."""

    lead = _finite(lead_full_year_log_return, field="lead full-year log return")
    raw = _strict_keys(
        comparator_full_year_log_returns,
        set(SIMPLE_COMPARATOR_NAMES),
        field="simple comparator returns",
    )
    comparator_returns = {
        name: _finite(raw[name], field=f"simple comparator return.{name}")
        for name in SIMPLE_COMPARATOR_NAMES
    }
    lead_minus = {
        name: float(math.fsum((lead, -value)))
        for name, value in comparator_returns.items()
    }
    criterion = _criterion_report(
        {
            f"lead_minus_{name}_gt_0_001": edge > EDGE_MATERIALITY
            for name, edge in lead_minus.items()
        }
    )
    return {
        **criterion,
        "cost_name": STRESS_COST_NAME,
        "lead_full_year_log_return": lead,
        "comparator_full_year_log_returns": comparator_returns,
        "lead_minus_comparator_log_returns": lead_minus,
    }


def classify_learning(
    *,
    economic_xor_fill_count: int,
    stress_complete_xor_episode_count: int,
    stress_distinct_xor_entry_quarter_count: int,
    stress_incremental_active_log_edge: float,
) -> str:
    """Return the mutually exclusive preregistered learning classification."""

    economic_rows = _exact_int(
        economic_xor_fill_count, field="economic XOR fill count"
    )
    episodes = _exact_int(
        stress_complete_xor_episode_count,
        field="stress complete XOR episode count",
    )
    quarters = _exact_int(
        stress_distinct_xor_entry_quarter_count,
        field="stress distinct XOR entry quarter count",
    )
    edge = _finite(
        stress_incremental_active_log_edge,
        field="stress incremental active log edge",
    )
    if economic_rows == 0:
        return "unexercised"
    if episodes < 3 or quarters < 2:
        return "exercised_insufficient_evidence"
    if edge > ZERO_TOLERANCE:
        return "exercised_positive"
    if edge < -ZERO_TOLERANCE:
        return "exercised_negative"
    return "exercised_flat"


def evaluate_learning_candidate(
    by_cost: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply learning candidacy without allowing it to affect stage success."""

    costs = _cost_mapping(by_cost, field="learning evidence")
    expected = {
        "economic_xor_fill_count",
        "complete_xor_episode_count",
        "distinct_xor_entry_quarters",
        "incremental_active_log_edge",
        "edge_after_removing_best_complete_xor_episode",
    }
    normalized: dict[str, Any] = {}
    for cost_name in COST_NAMES:
        value = _strict_keys(
            costs[cost_name], expected, field=f"learning evidence.{cost_name}"
        )
        quarters = value["distinct_xor_entry_quarters"]
        if (
            not isinstance(quarters, Sequence)
            or isinstance(quarters, (str, bytes))
            or any(name not in QUARTERS for name in quarters)
            or len(set(quarters)) != len(quarters)
            or list(quarters) != [name for name in QUARTERS if name in quarters]
        ):
            raise ContextualExpertAggregation2024AuditEvaluationError(
                f"learning evidence.{cost_name}.distinct_xor_entry_quarters "
                "must be unique and canonically ordered"
            )
        removed = value["edge_after_removing_best_complete_xor_episode"]
        normalized[cost_name] = {
            "economic_xor_fill_count": _exact_int(
                value["economic_xor_fill_count"],
                field=f"learning evidence.{cost_name}.economic_xor_fill_count",
            ),
            "complete_xor_episode_count": _exact_int(
                value["complete_xor_episode_count"],
                field=f"learning evidence.{cost_name}.complete_xor_episode_count",
            ),
            "distinct_xor_entry_quarters": list(quarters),
            "incremental_active_log_edge": _finite(
                value["incremental_active_log_edge"],
                field=f"learning evidence.{cost_name}.incremental_active_log_edge",
            ),
            "edge_after_removing_best_complete_xor_episode": (
                None
                if removed is None
                else _finite(
                    removed,
                    field=(
                        f"learning evidence.{cost_name}."
                        "edge_after_removing_best_complete_xor_episode"
                    ),
                )
            ),
        }
    economic_counts = {
        value["economic_xor_fill_count"] for value in normalized.values()
    }
    if len(economic_counts) != 1:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            "economic XOR fill count changed across costs"
        )

    stress = normalized[STRESS_COST_NAME]
    status = classify_learning(
        economic_xor_fill_count=stress["economic_xor_fill_count"],
        stress_complete_xor_episode_count=stress["complete_xor_episode_count"],
        stress_distinct_xor_entry_quarter_count=len(
            stress["distinct_xor_entry_quarters"]
        ),
        stress_incremental_active_log_edge=stress["incremental_active_log_edge"],
    )
    checks = {
        "complete_xor_episodes_at_least_3_both_costs": all(
            value["complete_xor_episode_count"] >= 3
            for value in normalized.values()
        ),
        "xor_entries_in_at_least_2_quarters_both_costs": all(
            len(value["distinct_xor_entry_quarters"]) >= 2
            for value in normalized.values()
        ),
        "incremental_active_log_edge_gt_0_001_both_costs": all(
            value["incremental_active_log_edge"] > EDGE_MATERIALITY
            for value in normalized.values()
        ),
        "stress_edge_after_removing_best_complete_xor_positive": (
            stress["edge_after_removing_best_complete_xor_episode"] is not None
            and stress["edge_after_removing_best_complete_xor_episode"] > 0.0
        ),
    }
    criterion = _criterion_report(checks)
    return {
        "classification": status,
        "learning_candidate_for_2025_shadow": criterion["passed"],
        "candidate_criterion": criterion,
        "by_cost": normalized,
    }


def aggregate_stage_status(
    *,
    aapl_policy_pass_by_cost: Mapping[str, bool],
    simple_policy_superiority_pass: bool,
    integrity_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Aggregate exact gates and choose the preregistered terminal status."""

    raw_costs = _cost_mapping(
        aapl_policy_pass_by_cost, field="AAPL policy pass statuses"
    )
    cost_flags = {
        cost: _exact_bool(raw_costs[cost], field=f"AAPL policy pass.{cost}")
        for cost in COST_NAMES
    }
    superiority = _exact_bool(
        simple_policy_superiority_pass, field="simple policy superiority pass"
    )
    if not isinstance(integrity_checks, Mapping) or not integrity_checks:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            "fatal integrity checks must be a nonempty mapping"
        )
    normalized_integrity: dict[str, bool] = {}
    for name, value in integrity_checks.items():
        if not isinstance(name, str) or not name:
            raise ContextualExpertAggregation2024AuditEvaluationError(
                "fatal integrity check names must be nonempty strings"
            )
        normalized_integrity[name] = _exact_bool(
            value, field=f"fatal integrity.{name}"
        )
    aapl_pass = all(cost_flags.values())
    integrity_pass = all(normalized_integrity.values())
    stage_pass = aapl_pass and superiority and integrity_pass
    if not integrity_pass:
        status = INTEGRITY_REJECTION_STATUS
    elif stage_pass:
        status = PASS_STATUS
    elif aapl_pass and not superiority:
        status = AAPL_ONLY_STATUS
    else:
        status = REJECTION_STATUS
    return {
        "aapl_policy_pass_by_cost": cost_flags,
        "aapl_policy_pass": aapl_pass,
        "simple_policy_superiority_pass": superiority,
        "fatal_integrity": {
            "passed": integrity_pass,
            "passed_count": sum(normalized_integrity.values()),
            "total_count": len(normalized_integrity),
            "checks": normalized_integrity,
            "failed_checks": [
                name for name, value in normalized_integrity.items() if not value
            ],
        },
        "stage_pass": stage_pass,
        "status": status,
    }


def _canonical_ledger(
    value: Any, *, cost_name: str, scenario_name: str
) -> pd.DataFrame:
    cost_bps = COST_BPS[cost_name]
    try:
        frame = _base._validated_ledger(
            value,
            field=f"{cost_name}.{scenario_name}.ledger",
            cost_bps=cost_bps,
            account_years=ACCOUNT_YEARS,
        )
    except _base.ContextualExpertAggregationEvaluationError as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(str(exc)) from exc
    observed = set(frame["policy_name"].tolist())
    expected = _INTERNAL_POLICY_NAMES[scenario_name]
    if observed != {expected}:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name}.{scenario_name} retained the wrong internal policy identity"
        )
    if not frame["post_fill_exposure"].equals(frame["requested_target_exposure"]):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name}.{scenario_name} post-fill exposure differs from its target"
        )
    return frame


def _exact_frame(
    value: Any, *, columns: Sequence[str], field: str
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or tuple(value.columns) != tuple(columns):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} does not have the exact inherited schema"
        )
    return value.copy()


def _validated_policy_evidence(
    value: Any, *, cost_name: str, scenario_name: str
) -> dict[str, Any]:
    evidence = _strict_keys(
        value,
        set(POLICY_EVIDENCE_KEYS),
        field=f"{cost_name}.{scenario_name}.policy evidence",
    )
    ledger = _canonical_ledger(
        evidence["strategy_ledger"],
        cost_name=cost_name,
        scenario_name=scenario_name,
    )
    supplied = evidence["cash_episodes"]
    if not isinstance(supplied, _evidence.CashEpisodeEvidence):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name}.{scenario_name}.cash_episodes must be CashEpisodeEvidence"
        )
    complete = _exact_frame(
        supplied.complete,
        columns=_ledger.EPISODE_COLUMNS,
        field=f"{cost_name}.{scenario_name}.complete episodes",
    )
    opened = _exact_frame(
        supplied.open,
        columns=_ledger.UNRESOLVED_EPISODE_COLUMNS,
        field=f"{cost_name}.{scenario_name}.open episodes",
    )
    start = _base._canonical_start_state(
        ledger,
        cost_bps=COST_BPS[cost_name],
        field=f"{cost_name}.{scenario_name}.ledger",
    )
    try:
        terminal_state = _ledger.verify_ledger(ledger, start_state=start)
        extracted = _evidence.cash_episode_evidence(
            ledger=ledger, terminal_state=terminal_state
        )
    except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name}.{scenario_name} CASH episode extraction failed"
        ) from exc
    except _evidence.AuditEvidenceError as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name}.{scenario_name} CASH episode extraction failed"
        ) from exc
    if (
        not extracted.complete.equals(complete)
        or not extracted.open.equals(opened)
        or dict(extracted.reconciliation) != dict(supplied.reconciliation)
    ):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name}.{scenario_name} CASH evidence differs from canonical extraction"
        )
    try:
        canonical_complete = _base._canonical_episode_frame(
            complete,
            cost_bps=COST_BPS[cost_name],
            field=f"{cost_name}.{scenario_name}.complete episodes",
        )
    except _base.ContextualExpertAggregationEvaluationError as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(str(exc)) from exc
    return {
        "ledger": ledger,
        "terminal_state": terminal_state,
        "complete_episodes": canonical_complete,
        "open_episodes": opened,
        "cash_reconciliation": dict(extracted.reconciliation),
    }


def _require_aligned_ledgers(
    left: pd.DataFrame, right: pd.DataFrame, *, field: str
) -> None:
    columns = (
        "row_index",
        "decision_date",
        "fill_date",
        "reference_adjusted_open",
        "cost_bps",
    )
    if not left.loc[:, columns].equals(right.loc[:, columns]):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} ledgers do not share exact fills, prices, and costs"
        )


def _period_mask(
    ledger: pd.DataFrame, *, months: Sequence[int] | None = None
) -> pd.Series:
    dates = pd.to_datetime(
        ledger["fill_date"], format="%Y-%m-%d", exact=True, errors="raise"
    )
    mask = dates.dt.year.eq(AUDIT_YEAR)
    if months is not None:
        mask &= dates.dt.month.isin(tuple(months))
    return mask


def _period_account_metrics(
    ledger: pd.DataFrame, mask: pd.Series, *, field: str
) -> dict[str, Any]:
    if not isinstance(mask, pd.Series) or not mask.index.equals(ledger.index):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} period mask is not aligned to the ledger"
        )
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))
    selected = ledger.loc[mask]
    if selected.empty:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} has no ledger fills"
        )
    if len(positions) > 1 and not np.array_equal(
        np.diff(positions), np.ones(len(positions) - 1, dtype=int)
    ):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} period mask is not one contiguous ledger interval"
        )
    first_position = int(positions[0])
    if first_position > 0:
        start_equity = _finite(
            ledger.iloc[first_position - 1]["equity"],
            field=f"{field}.start boundary equity",
        )
    else:
        first_end = _finite(
            selected.iloc[0]["equity"], field=f"{field}.first equity"
        )
        first_return = _finite(
            selected.iloc[0]["daily_return"],
            field=f"{field}.first daily return",
        )
        if first_return <= -1.0:
            raise ContextualExpertAggregation2024AuditEvaluationError(
                f"{field} first daily return destroys the account"
            )
        start_equity = _finite(
            first_end / (1.0 + first_return),
            field=f"{field}.start boundary equity",
        )
    end_equity = _finite(selected.iloc[-1]["equity"], field=f"{field}.end equity")
    boundary_log_return = math.log(end_equity / start_equity)
    atomic_log_return = float(
        math.fsum(
            math.log1p(_finite(value, field=f"{field}.daily return"))
            for value in selected["daily_return"].tolist()
        )
    )
    if abs(boundary_log_return - atomic_log_return) > RECONCILIATION_TOLERANCE:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} boundary return does not reconcile to daily returns"
        )
    peak = start_equity
    maximum_drawdown = 0.0
    for raw in selected["equity"].tolist():
        equity = _finite(raw, field=f"{field}.equity")
        peak = max(peak, equity)
        maximum_drawdown = min(maximum_drawdown, equity / peak - 1.0)
    return {
        "start_equity": start_equity,
        "end_equity": end_equity,
        "log_return": float(boundary_log_return),
        "simple_return": float(math.expm1(boundary_log_return)),
        "fill_count": len(selected),
        "trade_count": int(sum(value is True for value in selected["trade_executed"])),
        "target_change_count": int(
            sum(value is True for value in selected["target_changed"])
        ),
        "turnover_reference": float(
            math.fsum(
                _finite(value, field=f"{field}.turnover")
                for value in selected["turnover_reference"].tolist()
            )
        ),
        "maximum_drawdown": float(maximum_drawdown),
    }


def _relative_period_metrics(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    mask: pd.Series,
    *,
    field: str,
) -> dict[str, Any]:
    _require_aligned_ledgers(strategy, benchmark, field=field)
    strategy_metrics = _period_account_metrics(
        strategy, mask, field=f"{field}.strategy"
    )
    benchmark_metrics = _period_account_metrics(
        benchmark, mask, field=f"{field}.benchmark"
    )
    edge = float(
        math.fsum(
            (strategy_metrics["log_return"], -benchmark_metrics["log_return"])
        )
    )
    atomic = float(
        math.fsum(
            math.log1p(float(left)) - math.log1p(float(right))
            for left, right in zip(
                strategy.loc[mask, "daily_return"].tolist(),
                benchmark.loc[mask, "daily_return"].tolist(),
            )
        )
    )
    if abs(edge - atomic) > RECONCILIATION_TOLERANCE:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} active edge does not reconcile to daily increments"
        )
    return {
        "strategy": strategy_metrics,
        "aapl_buy_hold": benchmark_metrics,
        "active_log_edge": edge,
        "relative_wealth_advantage": float(math.expm1(edge)),
    }


def _open_episode_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    records = frame.to_dict(orient="records")
    return {
        "count": len(records),
        "statuses": [row["status"] for row in records],
        "net_active_log_edge_to_mark": float(
            math.fsum(float(row["net_active_log_edge_to_mark"]) for row in records)
        ),
    }


def _policy_report(
    policy: Mapping[str, Any], benchmark: Mapping[str, Any], *, cost_name: str
) -> dict[str, Any]:
    ledger = policy["ledger"]
    benchmark_ledger = benchmark["ledger"]
    year_mask = _period_mask(ledger)
    year = _relative_period_metrics(
        ledger, benchmark_ledger, year_mask, field=f"{cost_name}.calendar 2024"
    )
    quarters = {
        name: _relative_period_metrics(
            ledger,
            benchmark_ledger,
            _period_mask(ledger, months=months),
            field=f"{cost_name}.calendar 2024.{name}",
        )
        for name, months in QUARTERS.items()
    }
    if (
        abs(
            math.fsum(value["active_log_edge"] for value in quarters.values())
            - year["active_log_edge"]
        )
        > RECONCILIATION_TOLERANCE
    ):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name} quarter edges do not reconcile to the annual edge"
        )
    all_mask = pd.Series(True, index=ledger.index, dtype=bool)
    continuous = _relative_period_metrics(
        ledger,
        benchmark_ledger,
        all_mask,
        field=f"{cost_name}.continuous 2005-2024",
    )
    complete = policy["complete_episodes"]
    try:
        year_episodes = _base.summarize_complete_episodes(
            complete, years=(AUDIT_YEAR,), cost_bps=COST_BPS[cost_name]
        )
        continuous_episodes = _base.summarize_complete_episodes(
            complete, years=ACCOUNT_YEARS, cost_bps=COST_BPS[cost_name]
        )
    except _base.ContextualExpertAggregationEvaluationError as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(str(exc)) from exc
    opened = _open_episode_diagnostics(policy["open_episodes"])
    episode_total = float(
        math.fsum((continuous_episodes["sum"], opened["net_active_log_edge_to_mark"]))
    )
    if abs(episode_total - continuous["active_log_edge"]) > RECONCILIATION_TOLERANCE:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name} complete plus open CASH episodes do not reconcile"
        )
    return {
        "calendar_2024": {
            "full_year": year,
            "quarters": quarters,
            "complete_episode_statistics": year_episodes,
            "open_episode_diagnostics": opened,
        },
        "continuous_2005_2024": {
            **continuous,
            "complete_episode_statistics": continuous_episodes,
            "open_episode_diagnostics": opened,
            "complete_plus_open_episode_edge": episode_total,
        },
    }


def _require_policy_cross_cost_identity(
    by_cost: Mapping[str, Mapping[str, Any]], *, scenario_name: str
) -> None:
    try:
        _ledger.assert_cross_cost_action_identity(
            {
                cost: by_cost[cost][scenario_name]["ledger"]
                for cost in COST_NAMES
            }
        )
        _base._require_episode_identity(
            by_cost[BASE_COST_NAME][scenario_name]["complete_episodes"],
            by_cost[STRESS_COST_NAME][scenario_name]["complete_episodes"],
            field=scenario_name,
        )
    except (
        TypeError,
        ValueError,
        _ledger.BinaryLedgerError,
        _base.ContextualExpertAggregationEvaluationError,
    ) as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{scenario_name} action or episode identity changed across costs"
        ) from exc
    identity_columns = (
        "status",
        "entry_decision_date",
        "entry_fill_date",
        "entry_reference_price",
        "pending_decision_date",
        "pending_target_exposure",
        "mark_date",
        "mark_reference_price",
        "raw_active_log_edge_to_mark",
    )
    left = by_cost[BASE_COST_NAME][scenario_name]["open_episodes"]
    right = by_cost[STRESS_COST_NAME][scenario_name]["open_episodes"]
    if not left.loc[:, identity_columns].equals(right.loc[:, identity_columns]):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{scenario_name} open episode identity changed across costs"
        )


def _require_lead_shadow_prefix(
    lead: pd.DataFrame, shadow: pd.DataFrame, *, field: str
) -> None:
    prefix = lead["fill_date"] <= "2023-12-31"
    other_prefix = shadow["fill_date"] <= "2023-12-31"
    if not lead.loc[prefix].reset_index(drop=True).equals(
        shadow.loc[other_prefix].reset_index(drop=True)
    ):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} lead and shadow do not share the exact through-2023 account"
        )


def _mapped_orientation(value: Any, *, field: str) -> str:
    allowed = set(_GENERIC_TO_CONTRACT_ORIENTATION.values())
    if value not in allowed:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} has an invalid XOR orientation"
        )
    return str(value)


def _parent_state(policy: Mapping[str, Any], *, field: str) -> _ledger.AccountState:
    ledger = policy["ledger"]
    prefix = ledger.loc[ledger["fill_date"] <= "2023-12-31"].reset_index(drop=True)
    if prefix.empty or prefix.iloc[-1]["fill_date"] > "2023-12-31":
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} has no exact through-2023 account prefix"
        )
    start = _base._canonical_start_state(
        ledger,
        cost_bps=float(prefix.iloc[0]["cost_bps"]),
        field=field,
    )
    try:
        return _ledger.verify_ledger(prefix, start_state=start)
    except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{field} through-2023 account prefix failed replay"
        ) from exc


def _validated_xor_report(
    value: Any,
    *,
    cost_name: str,
    shadow: Mapping[str, Any],
    lead: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, _evidence.AuditXorEvidence):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name}.XOR evidence must be AuditXorEvidence"
        )
    components = _exact_frame(
        value.components,
        columns=_evidence.AUDIT_XOR_COMPONENT_COLUMNS,
        field=f"{cost_name}.XOR components",
    )
    complete = _exact_frame(
        value.complete,
        columns=_evidence.AUDIT_XOR_COLUMNS,
        field=f"{cost_name}.complete XOR episodes",
    )
    opened = _exact_frame(
        value.open,
        columns=_evidence.AUDIT_OPEN_XOR_COLUMNS,
        field=f"{cost_name}.open XOR episodes",
    )
    shadow_parent = _parent_state(shadow, field=f"{cost_name}.shadow")
    lead_parent = _parent_state(lead, field=f"{cost_name}.lead")
    try:
        extracted = _evidence.extract_audit_xor_evidence(
            full_shadow=shadow["ledger"],
            full_lead=lead["ledger"],
            shadow_parent_state=shadow_parent,
            lead_parent_state=lead_parent,
            shadow_terminal_state=shadow["terminal_state"],
            lead_terminal_state=lead["terminal_state"],
        )
    except (
        TypeError,
        ValueError,
        _ledger.BinaryLedgerError,
        _evidence.AuditEvidenceError,
    ) as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name} XOR extraction failed"
        ) from exc
    if (
        not extracted.components.equals(components)
        or not extracted.complete.equals(complete)
        or not extracted.open.equals(opened)
        or dict(extracted.reconciliation) != dict(value.reconciliation)
    ):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name} XOR evidence differs from canonical fill-based extraction"
        )
    economic_rows = components.loc[components["orientation_after"] != "equal"]
    full_period = _relative_period_metrics(
        shadow["ledger"],
        lead["ledger"],
        _period_mask(shadow["ledger"]),
        field=f"{cost_name}.shadow minus lead",
    )
    incremental_edge = full_period["active_log_edge"]
    component_edge = float(
        math.fsum(float(item) for item in components["net_incremental_log_edge"])
    )
    if abs(component_edge - incremental_edge) > RECONCILIATION_TOLERANCE:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name} XOR components do not reconcile to shadow-minus-lead edge"
        )

    complete_dates = (
        pd.to_datetime(
            complete["entry_fill_date"],
            format="%Y-%m-%d",
            exact=True,
            errors="raise",
        )
        if not complete.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    if not complete.empty and not complete_dates.dt.year.eq(AUDIT_YEAR).all():
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name} XOR divergence predates the exact shared prefix"
        )
    invalid_open = set(opened["status"].tolist()) - {"right_boundary_partial"}
    if invalid_open:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name} XOR evidence has a partial left boundary"
        )
    complete_edge = float(
        math.fsum(float(item) for item in complete["net_incremental_log_edge"])
    )
    open_edge = float(
        math.fsum(float(item) for item in opened["net_incremental_log_edge"])
    )
    if abs(math.fsum((complete_edge, open_edge)) - incremental_edge) > RECONCILIATION_TOLERANCE:
        raise ContextualExpertAggregation2024AuditEvaluationError(
            f"{cost_name} complete plus open XOR contributions do not reconcile"
        )
    entry_quarters = [
        name
        for name in QUARTERS
        if f"{AUDIT_YEAR}{name}" in set(complete["entry_quarter"].tolist())
    ]
    complete_values = [
        float(item) for item in complete["net_incremental_log_edge"].tolist()
    ]
    best = max(complete_values) if complete_values else None
    after_best = (
        None
        if best is None
        else float(math.fsum((incremental_edge, -best)))
    )
    try:
        statistics = _base._stats(complete_values, prefix="2024 XOR")
    except _base.ContextualExpertAggregationEvaluationError as exc:
        raise ContextualExpertAggregation2024AuditEvaluationError(str(exc)) from exc
    return {
        "incremental_active_log_edge": incremental_edge,
        "relative_wealth_advantage": full_period["relative_wealth_advantage"],
        "component_incremental_active_log_edge": component_edge,
        "complete_xor_incremental_active_log_edge": complete_edge,
        "open_xor_incremental_active_log_edge": open_edge,
        "reconciliation_error": float(
            math.fsum((incremental_edge, -complete_edge, -open_edge))
        ),
        "economic_xor_fill_count": len(economic_rows),
        "complete_xor_episode_count": len(complete),
        "open_xor_episode_count": len(opened),
        "distinct_xor_entry_quarters": entry_quarters,
        "edge_after_removing_best_complete_xor_episode": after_best,
        "complete_xor_statistics": statistics,
        "complete_orientations": sorted(
            {
                _mapped_orientation(value, field=f"{cost_name}.complete XOR")
                for value in complete["orientation"].tolist()
            }
        ),
        "open_orientations": sorted(
            {
                _mapped_orientation(value, field=f"{cost_name}.open XOR")
                for value in opened["orientation"].tolist()
            }
        ),
        "complete_xor_episode_diagnostics": complete.to_dict(orient="records"),
        "open_xor_episode_diagnostics": opened.to_dict(orient="records"),
        "evidence_reconciliation": dict(extracted.reconciliation),
    }


def _require_xor_cross_cost_identity(
    left: Any, right: Any
) -> None:
    if not isinstance(left, _evidence.AuditXorEvidence) or not isinstance(
        right, _evidence.AuditXorEvidence
    ):
        raise ContextualExpertAggregation2024AuditEvaluationError(
            "cross-cost XOR evidence must use AuditXorEvidence"
        )
    component_identity = (
        "fill_date",
        "reference_adjusted_open",
        "shadow_decision_date",
        "lead_decision_date",
        "shadow_before",
        "shadow_after",
        "lead_before",
        "lead_after",
        "orientation_after",
        "raw_market_component",
    )
    complete_identity = (
        "entry_fill_date",
        "exit_fill_date",
        "entry_quarter",
        "start_shadow_decision_date",
        "start_lead_decision_date",
        "end_shadow_decision_date",
        "end_lead_decision_date",
        "orientation",
        "xor_fill_observations",
        "raw_market_component",
    )
    open_identity = (
        "status",
        "entry_fill_date",
        "last_fill_date",
        "entry_quarter",
        "start_shadow_decision_date",
        "start_lead_decision_date",
        "last_shadow_decision_date",
        "last_lead_decision_date",
        "orientation",
        "xor_fill_observations",
        "raw_market_component",
    )
    pairs = (
        ("components", component_identity),
        ("complete", complete_identity),
        ("open", open_identity),
    )
    for name, columns in pairs:
        if not getattr(left, name).loc[:, columns].equals(
            getattr(right, name).loc[:, columns]
        ):
            raise ContextualExpertAggregation2024AuditEvaluationError(
                f"XOR {name} identity changed across costs"
            )


def evaluate_2024_audit(
    policy_evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    xor_evidence: Mapping[str, _evidence.AuditXorEvidence],
    integrity_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Validate primitives, compute metrics, and apply the full 2024 gate set.

    ``policy_evidence`` is keyed by cost and then by the six external scenario
    names.  Each scenario contains ``strategy_ledger`` and a canonical
    ``CashEpisodeEvidence`` value under ``cash_episodes``.  ``xor_evidence`` is
    keyed by cost and contains canonical ``AuditXorEvidence`` values.
    """

    raw_costs = _cost_mapping(policy_evidence, field="policy evidence")
    validated: dict[str, dict[str, Any]] = {}
    for cost_name in COST_NAMES:
        raw_policies = _strict_keys(
            raw_costs[cost_name],
            set(POLICY_NAMES),
            field=f"policy evidence.{cost_name}",
        )
        validated[cost_name] = {
            name: _validated_policy_evidence(
                raw_policies[name], cost_name=cost_name, scenario_name=name
            )
            for name in POLICY_NAMES
        }
        benchmark = validated[cost_name][AAPL_POLICY]["ledger"]
        for name in POLICY_NAMES:
            _require_aligned_ledgers(
                validated[cost_name][name]["ledger"],
                benchmark,
                field=f"{cost_name}.{name}",
            )
        _require_lead_shadow_prefix(
            validated[cost_name][LEAD_POLICY]["ledger"],
            validated[cost_name][SHADOW_POLICY]["ledger"],
            field=cost_name,
        )
        try:
            _ledger.assert_always_long_matches_buy_hold(
                validated[cost_name][ALWAYS_LONG_POLICY]["ledger"], benchmark
            )
        except (TypeError, ValueError, _ledger.BinaryLedgerError) as exc:
            raise ContextualExpertAggregation2024AuditEvaluationError(
                f"{cost_name} always-LONG economics differ from AAPL"
            ) from exc

    for scenario_name in POLICY_NAMES:
        _require_policy_cross_cost_identity(validated, scenario_name=scenario_name)

    policy_reports: dict[str, Any] = {}
    for cost_name in COST_NAMES:
        benchmark = validated[cost_name][AAPL_POLICY]
        policy_reports[cost_name] = {
            name: {
                "scenario_name": name,
                "internal_policy_name": _INTERNAL_POLICY_NAMES[name],
                **_policy_report(
                    validated[cost_name][name], benchmark, cost_name=cost_name
                ),
            }
            for name in POLICY_NAMES
        }

    raw_xor = _cost_mapping(xor_evidence, field="XOR evidence")
    _require_xor_cross_cost_identity(
        raw_xor[BASE_COST_NAME],
        raw_xor[STRESS_COST_NAME],
    )
    xor_reports = {
        cost_name: _validated_xor_report(
            raw_xor[cost_name],
            cost_name=cost_name,
            shadow=validated[cost_name][SHADOW_POLICY],
            lead=validated[cost_name][LEAD_POLICY],
        )
        for cost_name in COST_NAMES
    }

    aapl_gates: dict[str, Any] = {}
    for cost_name in COST_NAMES:
        lead = policy_reports[cost_name][LEAD_POLICY]["calendar_2024"]
        aapl_gates[cost_name] = evaluate_aapl_policy_gate(
            full_year_active_log_edge=lead["full_year"]["active_log_edge"],
            quarter_active_log_edges={
                name: lead["quarters"][name]["active_log_edge"]
                for name in QUARTERS
            },
            aapl_quarter_log_returns={
                name: lead["quarters"][name]["aapl_buy_hold"]["log_return"]
                for name in QUARTERS
            },
        )

    stress_policies = policy_reports[STRESS_COST_NAME]
    superiority = evaluate_simple_policy_superiority(
        lead_full_year_log_return=stress_policies[LEAD_POLICY]["calendar_2024"][
            "full_year"
        ]["strategy"]["log_return"],
        comparator_full_year_log_returns={
            name: stress_policies[name]["calendar_2024"]["full_year"]["strategy"][
                "log_return"
            ]
            for name in SIMPLE_COMPARATOR_NAMES
        },
    )

    learning_input = {
        cost_name: {
            name: xor_reports[cost_name][name]
            for name in (
                "economic_xor_fill_count",
                "complete_xor_episode_count",
                "distinct_xor_entry_quarters",
                "incremental_active_log_edge",
                "edge_after_removing_best_complete_xor_episode",
            )
        }
        for cost_name in COST_NAMES
    }
    learning = evaluate_learning_candidate(learning_input)
    stage = aggregate_stage_status(
        aapl_policy_pass_by_cost={
            cost_name: aapl_gates[cost_name]["passed"] for cost_name in COST_NAMES
        },
        simple_policy_superiority_pass=superiority["passed"],
        integrity_checks=integrity_checks,
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "metrics": {
            "policy_order": list(POLICY_NAMES),
            "policy_by_cost": policy_reports,
            "online_shadow_minus_frozen_lead_by_cost": xor_reports,
            "learning": learning,
        },
        "gate_report": {
            "edge_materiality": EDGE_MATERIALITY,
            "reconciliation_tolerance": RECONCILIATION_TOLERANCE,
            "zero_classification_tolerance": ZERO_TOLERANCE,
            "aapl_policy_by_cost": aapl_gates,
            "simple_policy_superiority": superiority,
            "learning_classification": learning["classification"],
            "learning_candidate_for_2025_shadow": learning[
                "learning_candidate_for_2025_shadow"
            ],
            **stage,
        },
    }
    json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return report


__all__ = [
    "ACCOUNT_YEARS",
    "AAPL_ONLY_STATUS",
    "AAPL_POLICY",
    "ALWAYS_LONG_POLICY",
    "AUDIT_YEAR",
    "BASE_COST_NAME",
    "CONTEXTUAL_POLICY",
    "CONTRACT_VERSION",
    "COST_BPS",
    "COST_NAMES",
    "ContextualExpertAggregation2024AuditEvaluationError",
    "EDGE_MATERIALITY",
    "EVIDENCE_CLASSIFICATION",
    "EXACT_UNION_POLICY",
    "INTEGRITY_REJECTION_STATUS",
    "LEAD_POLICY",
    "LEARNING_STATUSES",
    "PASS_STATUS",
    "POLICY_EVIDENCE_KEYS",
    "POLICY_NAMES",
    "QUARTERS",
    "RECONCILIATION_TOLERANCE",
    "REJECTION_STATUS",
    "SHADOW_POLICY",
    "SIMPLE_COMPARATOR_NAMES",
    "STRESS_COST_NAME",
    "ZERO_TOLERANCE",
    "aggregate_stage_status",
    "classify_learning",
    "evaluate_2024_audit",
    "evaluate_aapl_policy_gate",
    "evaluate_learning_candidate",
    "evaluate_simple_policy_superiority",
]
