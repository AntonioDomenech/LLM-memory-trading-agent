from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd
import pytest

import agent_benchmark.contextual_expert_aggregation_audit_evaluation as audit
from agent_benchmark import contextual_expert_aggregation_ledger as ledger


def _market(
    years: Sequence[int],
    *,
    episodes_per_year: int = 7,
    negative_years: frozenset[int] = frozenset({2008, 2022}),
) -> tuple[pd.Series, pd.Series]:
    """Synthetic prices only; no repository market-data artifact is opened."""

    dates: list[pd.Timestamp] = []
    prices: list[float] = []
    targets: list[int] = []
    for year in years:
        year_dates = pd.date_range(
            f"{year}-01-03", periods=2 * episodes_per_year + 2, freq="14D"
        )
        year_prices = [100.0]
        year_targets: list[int] = []
        for _ in range(episodes_per_year):
            year_targets.extend((0, 1))
            year_prices.extend((100.0, 98.0))
        year_targets.extend((1, 1))
        year_prices.append(90.0 if year in negative_years else 100.0)
        dates.extend(year_dates.tolist())
        prices.extend(year_prices)
        targets.extend(year_targets)
    index = pd.DatetimeIndex(dates)
    return (
        pd.Series(prices, index=index, dtype=float),
        pd.Series(targets, index=index, dtype=int),
    )


def _run(
    opens: pd.Series,
    targets: pd.Series,
    *,
    policy_name: str,
    cost_name: str,
) -> pd.DataFrame:
    return ledger.run_continuous_ledger(
        opens,
        targets,
        policy_name=policy_name,
        cost_bps=audit.COST_BPS[cost_name],
    ).ledger


def _episodes(frame: pd.DataFrame, *, cost_name: str) -> pd.DataFrame:
    state = ledger.AccountState.initial(
        policy_name=str(frame.iloc[0]["policy_name"]),
        cost_bps=audit.COST_BPS[cost_name],
    )
    extraction = ledger.extract_cash_episodes(frame, start_state=state)
    assert extraction.unresolved.empty
    return extraction.complete


def _xor(
    primary: pd.DataFrame, comparator: pd.DataFrame, *, cost_name: str
) -> pd.DataFrame:
    primary_state = ledger.AccountState.initial(
        policy_name=str(primary.iloc[0]["policy_name"]),
        cost_bps=audit.COST_BPS[cost_name],
    )
    comparator_state = ledger.AccountState.initial(
        policy_name=str(comparator.iloc[0]["policy_name"]),
        cost_bps=audit.COST_BPS[cost_name],
    )
    extraction = ledger.extract_signed_xor_episodes(
        primary,
        comparator,
        primary_start_state=primary_state,
        comparator_start_state=comparator_state,
    )
    assert extraction.unresolved.empty
    return extraction.complete


def _policy(
    opens: pd.Series,
    targets: pd.Series,
    *,
    policy_name: str,
    cost_name: str,
) -> dict[str, pd.DataFrame]:
    strategy = _run(
        opens, targets, policy_name=policy_name, cost_name=cost_name
    )
    benchmark = _run(
        opens,
        pd.Series(1, index=targets.index, dtype=int),
        policy_name="buy_hold",
        cost_name=cost_name,
    )
    return {
        "strategy_ledger": strategy,
        "benchmark_ledger": benchmark,
        "complete_episodes": _episodes(strategy, cost_name=cost_name),
    }


def _pairwise(
    opens: pd.Series,
    primary_targets: pd.Series,
    comparator_targets: pd.Series,
    *,
    comparator_name: str,
    cost_name: str,
) -> dict[str, pd.DataFrame]:
    primary = _run(
        opens, primary_targets, policy_name="online_full", cost_name=cost_name
    )
    comparator = _run(
        opens,
        comparator_targets,
        policy_name=comparator_name,
        cost_name=cost_name,
    )
    benchmark = _run(
        opens,
        pd.Series(1, index=primary_targets.index, dtype=int),
        policy_name="buy_hold",
        cost_name=cost_name,
    )
    return {
        "strategy_ledger": comparator,
        "benchmark_ledger": benchmark,
        "complete_episodes": _episodes(comparator, cost_name=cost_name),
        "learner_minus_comparator_xor": _xor(
            primary, comparator, cost_name=cost_name
        ),
    }


def _state_frame(
    online_targets: pd.Series, comparator_targets: pd.Series
) -> pd.DataFrame:
    online = online_targets.astype(int).tolist()
    comparator = comparator_targets.astype(int).tolist()
    return pd.DataFrame(
        {
            "decision_date": online_targets.index.strftime("%Y-%m-%d").tolist(),
            "canonical_union_opportunity": [True] * len(online),
            "online_cash_score": [0.8 if target == 0 else 0.2 for target in online],
            "comparator_cash_score": [
                0.8 if target == 0 else 0.2 for target in comparator
            ],
            "online_action": ["CASH" if target == 0 else "LONG" for target in online],
            "comparator_action": [
                "CASH" if target == 0 else "LONG" for target in comparator
            ],
        },
        columns=audit.ADAPTIVE_STATE_COLUMNS,
    )


def _evidence(
    *, negative_years: frozenset[int] = frozenset({2008, 2022})
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, dict[str, pd.DataFrame]]],
    dict[str, pd.DataFrame],
]:
    opens, learner_targets = _market(
        audit.CONTINUOUS_YEARS, negative_years=negative_years
    )
    always_long = pd.Series(1, index=learner_targets.index, dtype=int)

    frozen = learner_targets.copy()
    frozen.loc[frozen.index.year >= 2019] = 1
    global_only = learner_targets.copy()
    lifetime_only = learner_targets.copy()
    lifetime_only.loc[lifetime_only.index.year == 2019] = 1

    comparator_targets = {
        "always_long": always_long,
        "exact_union_cash": always_long,
        "contextual_only": always_long,
        "weak_trend_only": always_long,
        "online_minus_frozen_2018": frozen,
        "full_minus_global_only": global_only,
        "full_minus_lifetime_only": lifetime_only,
    }
    comparator_policy_names = {
        "online_minus_frozen_2018": "frozen_2018",
        "full_minus_global_only": "global_only",
        "full_minus_lifetime_only": "lifetime_only",
        **{name: name for name in audit.FIXED_COMPARATOR_NAMES},
    }
    policy = {
        cost: _policy(
            opens,
            learner_targets,
            policy_name="online_full",
            cost_name=cost,
        )
        for cost in audit.COST_NAMES
    }
    comparators = {
        cost: {
            name: _pairwise(
                opens,
                learner_targets,
                targets,
                comparator_name=comparator_policy_names[name],
                cost_name=cost,
            )
            for name, targets in comparator_targets.items()
        }
        for cost in audit.COST_NAMES
    }
    states = {
        name: _state_frame(learner_targets, comparator_targets[name])
        for name in audit.ADAPTIVE_COMPARISON_NAMES
    }
    return policy, comparators, states


@pytest.fixture(scope="module")
def passing_evidence() -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, dict[str, pd.DataFrame]]],
    dict[str, pd.DataFrame],
]:
    return _evidence()


@pytest.fixture(scope="module")
def passing_report(passing_evidence) -> dict[str, object]:
    policy, comparators, states = passing_evidence
    return audit.apply_audit_gates(
        policy,
        comparator_evidence=comparators,
        adaptive_state_differences=states,
    )


def test_synthetic_policy_passes_both_costs_and_preserves_tail_metrics(
    passing_report: dict[str, object],
) -> None:
    report = passing_report
    statuses = report["decision_statuses"]
    assert statuses["post_rejection_2019_2023_pass"] is True
    assert statuses["continuous_2005_2023_robustness_pass"] is True
    assert statuses["historical_policy_candidate_for_2024_audit"] is True
    assert statuses["learning_candidate_for_2024_audit"] is True

    for cost_name in audit.COST_NAMES:
        policy = report["policy_by_cost"][cost_name]
        suffix = policy["suffix_2019_2023"]
        continuous = policy["continuous_2005_2023"]
        assert suffix["criterion"]["passed"] is True
        assert suffix["positive_block_count"] == 3
        assert suffix["edge_after_removing_best_year"] > 0.0
        assert suffix["negative_aapl_years"] == [2022]
        assert suffix["negative_aapl_year_support_status"] == "observed_positive"
        assert suffix["episodes"]["positive_concentration"] <= 0.50

        assert continuous["criterion"]["passed"] is True
        assert continuous["episodes"]["count"] == 7 * 19
        assert continuous["edge_after_removing_five_largest_episodes"] > 0.0
        assert continuous["episodes"]["positive_concentration"] <= 0.25
        # The same-ledger benchmark can also make inception year 2005
        # slightly negative after its initial purchase cost.
        assert {2008, 2022}.issubset(continuous["negative_aapl_years"])
        assert continuous["negative_aapl_year_edge_sum"] > 0.0
        assert continuous["strategy_max_drawdown"] >= continuous["aapl_max_drawdown"]


def test_all_arms_and_fixed_policies_report_every_year_and_suffix_block(
    passing_report: dict[str, object],
) -> None:
    diagnostics = passing_report["all_policy_diagnostics"]
    assert diagnostics["policy_order"] == list(audit.AUDIT_POLICY_NAMES)
    for cost_name in audit.COST_NAMES:
        cost = diagnostics["by_cost"][cost_name]
        assert cost["policy_order"] == list(audit.AUDIT_POLICY_NAMES)
        assert set(cost["policies"]) == set(audit.AUDIT_POLICY_NAMES)
        for policy_name in audit.AUDIT_POLICY_NAMES:
            policy = cost["policies"][policy_name]
            assert policy["policy_name"] == policy_name
            assert set(policy["suffix_2019_2023"]["annual"]) == {
                str(year) for year in audit.SUFFIX_YEARS
            }
            assert set(policy["suffix_2019_2023"]["fixed_block_edges"]) == set(
                audit.SUFFIX_BLOCKS
            )
            assert set(policy["continuous_2005_2023"]["annual"]) == {
                str(year) for year in audit.CONTINUOUS_YEARS
            }


def test_negative_suffix_year_absence_is_explicitly_not_applicable() -> None:
    opens, targets = _market(
        audit.CONTINUOUS_YEARS, negative_years=frozenset({2008})
    )
    policy = {
        cost: _policy(
            opens, targets, policy_name="online_full", cost_name=cost
        )
        for cost in audit.COST_NAMES
    }
    evaluated = audit._base._evaluate_primary_by_cost(
        policy, stage="confirmation"
    )
    for cost_name in audit.COST_NAMES:
        suffix = audit._policy_cost_report(
            evaluated[cost_name], cost_name=cost_name
        )["suffix_2019_2023"]
        assert suffix["negative_aapl_years"] == []
        assert suffix["negative_aapl_year_support_status"] == (
            "not_applicable_no_negative_aapl_years"
        )
        assert suffix["criterion"]["checks"][
            "negative_aapl_year_edge_positive_if_applicable"
        ] is None
        assert suffix["criterion"]["not_applicable_checks"] == [
            "negative_aapl_year_edge_positive_if_applicable"
        ]
        assert suffix["criterion"]["passed"] is True


@pytest.mark.parametrize(
    ("score_differences", "crossings", "episodes", "years", "edge", "expected"),
    [
        (0, 0, 0, 0, 1.0, "unexercised"),
        (3, 0, 0, 0, 1.0, "unexercised"),
        (3, 3, 4, 2, 1.0, "exercised_insufficient_evidence"),
        (3, 3, 5, 1, 1.0, "exercised_insufficient_evidence"),
        (3, 3, 5, 2, 1e-6, "exercised_positive"),
        (3, 3, 5, 2, -1e-6, "exercised_negative"),
        (3, 3, 5, 2, 1e-12, "exercised_flat"),
        (3, 3, 5, 2, -1e-12, "exercised_flat"),
    ],
)
def test_adaptive_status_boundaries(
    score_differences: int,
    crossings: int,
    episodes: int,
    years: int,
    edge: float,
    expected: str,
) -> None:
    assert (
        audit.classify_adaptive_status(
            cash_score_difference_count=score_differences,
            threshold_crossing_count=crossings,
            xor_episode_count=episodes,
            distinct_xor_entry_year_count=years,
            stress_incremental_edge=edge,
        )
        == expected
    )


def test_all_three_adaptive_comparisons_are_classified_from_synthetic_state(
    passing_report: dict[str, object],
) -> None:
    adaptive = passing_report["adaptive_value"]
    assert adaptive["online_minus_frozen_2018"]["adaptive_status"] == (
        "exercised_positive"
    )
    assert adaptive["full_minus_global_only"]["adaptive_status"] == "unexercised"
    assert adaptive["full_minus_lifetime_only"]["adaptive_status"] == (
        "exercised_insufficient_evidence"
    )
    for name in audit.ADAPTIVE_COMPARISON_NAMES:
        item = adaptive[name]
        suffix_state = item["state_differences"]["suffix_2019_2023"]
        assert suffix_state["threshold_crossing_count"] == suffix_state[
            "action_xor_decision_count"
        ]
        assert set(item["by_cost"]) == set(audit.COST_NAMES)


def test_combined_status_requires_both_costs_and_all_learning_conditions() -> None:
    mixed = audit.aggregate_decision_statuses(
        post_rejection_by_cost={"base_5bps": True, "stress_10bps": False},
        continuous_by_cost={"base_5bps": True, "stress_10bps": True},
        online_minus_frozen_suffix_edges={"base_5bps": 0.1, "stress_10bps": 0.1},
        online_minus_union_suffix_edges={"base_5bps": 0.1, "stress_10bps": 0.1},
        online_minus_frozen_adaptive_status="exercised_positive",
    )
    assert mixed["post_rejection_2019_2023_pass"] is False
    assert mixed["historical_policy_candidate_for_2024_audit"] is False
    assert mixed["learning_candidate_for_2024_audit"] is False

    no_union_superiority = audit.aggregate_decision_statuses(
        post_rejection_by_cost={"base_5bps": True, "stress_10bps": True},
        continuous_by_cost={"base_5bps": True, "stress_10bps": True},
        online_minus_frozen_suffix_edges={"base_5bps": 0.1, "stress_10bps": 0.1},
        online_minus_union_suffix_edges={"base_5bps": 0.1, "stress_10bps": 0.0},
        online_minus_frozen_adaptive_status="exercised_positive",
    )
    assert no_union_superiority["historical_policy_candidate_for_2024_audit"] is True
    assert no_union_superiority["learning_candidate_for_2024_audit"] is False


def test_concentration_and_top_five_arithmetic_do_not_round_away_dominance() -> None:
    values = [9.0, 0.25, 0.25, 0.25, 0.25, -0.5]
    concentration = audit._base._positive_concentration(values)
    assert math.isclose(concentration, 0.9, rel_tol=0.0, abs_tol=1e-15)
    assert concentration > 0.50
    assert math.isclose(
        audit._top_k_removed(values, k=5), -0.5, rel_tol=0.0, abs_tol=1e-15
    )


def test_state_actions_must_recompute_from_frozen_threshold(
    passing_evidence,
) -> None:
    policy, comparators, states = passing_evidence
    tampered = states["online_minus_frozen_2018"].copy()
    tampered.loc[tampered.index[-1], "online_action"] = "CASH"
    states["online_minus_frozen_2018"] = tampered
    with pytest.raises(
        audit.ContextualExpertAggregationAuditEvaluationError,
        match="does not match the frozen threshold",
    ):
        audit._canonical_state_frame(
            states["online_minus_frozen_2018"],
            primary_ledger=policy["stress_10bps"]["strategy_ledger"],
            comparator_ledger=comparators["stress_10bps"][
                "online_minus_frozen_2018"
            ]["strategy_ledger"],
            field="tampered synthetic state",
        )
