from __future__ import annotations

import math

import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_2024_audit_evaluation as evaluation


def _quarters(*values: float) -> dict[str, float]:
    return dict(zip(evaluation.QUARTERS, values, strict=True))


def _learning_cost(
    *,
    fills: int = 5,
    episodes: int = 3,
    quarters: tuple[str, ...] = ("Q1", "Q2"),
    edge: float = 0.002,
    removed: float | None = 0.0001,
) -> dict[str, object]:
    return {
        "economic_xor_fill_count": fills,
        "complete_xor_episode_count": episodes,
        "distinct_xor_entry_quarters": list(quarters),
        "incremental_active_log_edge": edge,
        "edge_after_removing_best_complete_xor_episode": removed,
    }


def test_aapl_gate_is_strict_unrounded_and_handles_not_applicable_support() -> None:
    exact = evaluation.evaluate_aapl_policy_gate(
        full_year_active_log_edge=0.001,
        quarter_active_log_edges=_quarters(0.001, 0.001, -0.0005, -0.0005),
        aapl_quarter_log_returns=_quarters(0.1, 0.1, 0.1, 0.1),
    )
    assert exact["passed"] is False
    assert exact["checks"]["full_year_active_log_edge_gt_0_001"] is False
    assert exact["checks"]["negative_aapl_quarter_edge_positive_if_applicable"] is None
    assert exact["negative_aapl_quarter_support_status"] == (
        "not_applicable_no_negative_aapl_quarters"
    )

    passing = evaluation.evaluate_aapl_policy_gate(
        full_year_active_log_edge=0.004,
        quarter_active_log_edges=_quarters(0.002, 0.001, 0.001, 0.0),
        aapl_quarter_log_returns=_quarters(0.1, -0.01, 0.1, 0.1),
    )
    assert passing["passed"] is True
    assert passing["negative_aapl_quarters"] == ["Q2"]


def test_simple_superiority_does_not_turn_equality_into_a_pass() -> None:
    report = evaluation.evaluate_simple_policy_superiority(
        lead_full_year_log_return=0.001,
        comparator_full_year_log_returns={
            evaluation.CONTEXTUAL_POLICY: 0.0,
            evaluation.EXACT_UNION_POLICY: -0.001,
        },
    )
    assert report["lead_minus_comparator_log_returns"][
        evaluation.CONTEXTUAL_POLICY
    ] == pytest.approx(0.001)
    assert report["passed"] is False


@pytest.mark.parametrize(
    ("fills", "episodes", "quarters", "edge", "expected"),
    [
        (0, 0, (), 1.0, "unexercised"),
        (1, 2, ("Q1", "Q2"), 1.0, "exercised_insufficient_evidence"),
        (1, 3, ("Q1",), 1.0, "exercised_insufficient_evidence"),
        (1, 3, ("Q1", "Q2"), 1.0001e-12, "exercised_positive"),
        (1, 3, ("Q1", "Q2"), -1.0001e-12, "exercised_negative"),
        (1, 3, ("Q1", "Q2"), 1e-12, "exercised_flat"),
        (1, 3, ("Q1", "Q2"), -1e-12, "exercised_flat"),
    ],
)
def test_learning_classification_boundaries(
    fills: int,
    episodes: int,
    quarters: tuple[str, ...],
    edge: float,
    expected: str,
) -> None:
    assert (
        evaluation.classify_learning(
            economic_xor_fill_count=fills,
            stress_complete_xor_episode_count=episodes,
            stress_distinct_xor_entry_quarter_count=len(quarters),
            stress_incremental_active_log_edge=edge,
        )
        == expected
    )


def test_learning_candidate_requires_every_preregistered_gate_at_both_costs() -> None:
    passing = evaluation.evaluate_learning_candidate(
        {name: _learning_cost() for name in evaluation.COST_NAMES}
    )
    assert passing["classification"] == "exercised_positive"
    assert passing["learning_candidate_for_2025_shadow"] is True

    equality = {
        name: _learning_cost(edge=0.001) for name in evaluation.COST_NAMES
    }
    report = evaluation.evaluate_learning_candidate(equality)
    assert report["learning_candidate_for_2025_shadow"] is False


def test_stage_status_precedence_and_learning_independence() -> None:
    passing_costs = {name: True for name in evaluation.COST_NAMES}
    assert evaluation.aggregate_stage_status(
        aapl_policy_pass_by_cost=passing_costs,
        simple_policy_superiority_pass=True,
        integrity_checks={"causal": True},
    ) == {
        "aapl_policy_pass_by_cost": passing_costs,
        "aapl_policy_pass": True,
        "simple_policy_superiority_pass": True,
        "fatal_integrity": {
            "passed": True,
            "passed_count": 1,
            "total_count": 1,
            "checks": {"causal": True},
            "failed_checks": [],
        },
        "stage_pass": True,
        "status": evaluation.PASS_STATUS,
    }
    assert evaluation.aggregate_stage_status(
        aapl_policy_pass_by_cost=passing_costs,
        simple_policy_superiority_pass=False,
        integrity_checks={"causal": True},
    )["status"] == evaluation.AAPL_ONLY_STATUS
    assert evaluation.aggregate_stage_status(
        aapl_policy_pass_by_cost=passing_costs,
        simple_policy_superiority_pass=True,
        integrity_checks={"causal": False},
    )["status"] == evaluation.INTEGRITY_REJECTION_STATUS


def test_period_metrics_use_the_prior_ledger_equity_boundary() -> None:
    ledger = pd.DataFrame(
        {
            "equity_before_fill": [1.1, 1.25, 1.05],
            "equity": [1.1, 1.21, 1.089],
            "daily_return": [0.1, 0.1, -0.1],
            "trade_executed": [False, False, False],
            "target_changed": [False, False, False],
            "turnover_reference": [0.0, 0.0, 0.0],
        }
    )
    mask = pd.Series([False, True, True], index=ledger.index)

    report = evaluation._period_account_metrics(
        ledger, mask, field="synthetic boundary"
    )

    assert report["start_equity"] == pytest.approx(1.1)
    assert report["end_equity"] == pytest.approx(1.089)
    assert report["log_return"] == pytest.approx(math.log(0.99))

    with pytest.raises(
        evaluation.ContextualExpertAggregation2024AuditEvaluationError,
        match="contiguous",
    ):
        evaluation._period_account_metrics(
            ledger,
            pd.Series([True, False, True], index=ledger.index),
            field="synthetic noncontiguous",
        )
