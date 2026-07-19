from __future__ import annotations

import pandas as pd

from agent_benchmark.regime_expert_disagreement_experiment import (
    apply_development_gates,
    build_calibration_diagnostics,
    build_policy_targets,
)


def _rows(regime: str, expert: str, edges: list[float]) -> list[dict[str, object]]:
    return [
        {
            "regime": regime,
            "expert_membership": expert,
            "entry_year": 2005 + (index % 4),
            "net_cash_log_edge_10bps": edge,
        }
        for index, edge in enumerate(edges)
    ]


def test_calibration_selects_best_eligible_expert_and_rejects_sparse_cell() -> None:
    rows = [
        *_rows("risk_on", "contextual_only", [0.03, 0.02, 0.01, 0.02, 0.01]),
        *_rows("risk_on", "weak_trend_only", [0.01, 0.01, 0.01, 0.01, 0.01]),
        *_rows("not_risk_on", "contextual_only", [0.04, 0.04, 0.04, 0.04]),
        *_rows("not_risk_on", "weak_trend_only", [-0.01] * 6),
    ]

    diagnostics, choices = build_calibration_diagnostics(rows)

    assert choices == {"risk_on": "contextual_only", "not_risk_on": None}
    assert diagnostics["risk_on"]["contextual_only"]["eligible"] is True
    assert diagnostics["not_risk_on"]["contextual_only"]["eligible"] is False
    assert diagnostics["not_risk_on"]["weak_trend_only"]["eligible"] is False


def test_calibration_exact_mean_tie_prefers_contextual() -> None:
    rows = [
        *_rows("risk_on", "contextual_only", [0.01] * 5),
        *_rows("risk_on", "weak_trend_only", [0.01] * 5),
    ]

    _, choices = build_calibration_diagnostics(rows)

    assert choices["risk_on"] == "contextual_only"


def test_policy_keeps_pre2012_union_and_both_but_vetoes_unselected_expert() -> None:
    dates = pd.bdate_range("2011-12-01", "2012-02-29")
    frame = pd.DataFrame(index=dates)
    membership = pd.DataFrame(
        False,
        index=dates,
        columns=("union", "both", "contextual_only", "weak_trend_only"),
    )
    pre = dates.get_loc(pd.Timestamp("2011-12-20"))
    contextual = dates.get_loc(pd.Timestamp("2012-01-10"))
    weak = dates.get_loc(pd.Timestamp("2012-01-20"))
    both = dates.get_loc(pd.Timestamp("2012-02-03"))
    membership.iloc[pre, membership.columns.get_loc("union")] = True
    membership.iloc[pre, membership.columns.get_loc("weak_trend_only")] = True
    membership.iloc[contextual, membership.columns.get_loc("union")] = True
    membership.iloc[contextual, membership.columns.get_loc("contextual_only")] = True
    membership.iloc[weak, membership.columns.get_loc("union")] = True
    membership.iloc[weak, membership.columns.get_loc("weak_trend_only")] = True
    membership.iloc[both, membership.columns.get_loc("union")] = True
    membership.iloc[both, membership.columns.get_loc("both")] = True
    regime = pd.DataFrame(
        {"regime_ready": True, "risk_on": True}, index=dates
    )

    targets, decisions = build_policy_targets(
        frame,
        membership,
        regime,
        {"risk_on": "contextual_only", "not_risk_on": None},
    )

    assert targets["selector"].iloc[pre] == 0.0
    assert targets["selector"].iloc[contextual] == 0.0
    assert targets["selector"].iloc[weak] == 1.0
    assert targets["selector"].iloc[both] == 0.0
    assert (targets["selector"] >= targets["union"]).all()
    assert decisions.iloc[weak]["reason"].endswith("vetoed")


def test_policy_uses_entry_boundary_and_masks_incomplete_tail() -> None:
    dates = pd.bdate_range("2004-12-29", "2005-01-12")
    frame = pd.DataFrame(index=dates)
    membership = pd.DataFrame(
        False,
        index=dates,
        columns=("union", "both", "contextual_only", "weak_trend_only"),
    )
    first_2005_entry = dates.get_loc(pd.Timestamp("2004-12-31"))
    incomplete_tail = len(dates) - 2
    for position in (first_2005_entry, incomplete_tail):
        membership.iloc[position, membership.columns.get_loc("union")] = True
        membership.iloc[position, membership.columns.get_loc("both")] = True
    regime = pd.DataFrame(
        {"regime_ready": True, "risk_on": True}, index=dates
    )

    targets, _ = build_policy_targets(
        frame,
        membership,
        regime,
        {"risk_on": "contextual_only", "not_risk_on": None},
    )

    assert dates[first_2005_entry] < pd.Timestamp("2005-01-01")
    assert dates[first_2005_entry + 1] >= pd.Timestamp("2005-01-01")
    assert targets["union"].iloc[first_2005_entry] == 0.0
    assert targets["selector"].iloc[first_2005_entry] == 0.0
    assert targets["union"].iloc[incomplete_tail] == 1.0
    assert targets["selector"].iloc[incomplete_tail] == 1.0


def _policy_metrics(
    *, edge: float, episodes: int = 30, mean: float = 0.01, median: float = 0.01
) -> dict[str, object]:
    return {
        "total_active_log_edge": edge,
        "cash_episode_count": episodes,
        "mean_cash_episode_edge": mean,
        "median_cash_episode_edge": median,
        "no_leverage_proof": {"passed": True},
        "comparison": {},
    }


def test_development_gates_pass_and_name_real_trading_failure() -> None:
    incremental = {
        "total_incremental_active_log_edge": 0.07,
        "positive_incremental_year_count": 7,
        "incremental_after_best_year_removed": 0.05,
    }
    metrics = {
        cost: {
            "selector": _policy_metrics(edge=0.5),
            "union": _policy_metrics(edge=0.43),
            "always_long": _policy_metrics(edge=0.0, episodes=0),
            "selector_vs_union": dict(incremental),
        }
        for cost in ("base_5bps", "stress_10bps")
    }
    metrics["integrity"] = {
        "selector_cash_subset_of_union": True,
        "benchmark_identical_across_policies": True,
        "always_long_equals_buy_and_hold": True,
    }

    passing = apply_development_gates(metrics)
    assert passing["passed"] is True

    metrics["stress_10bps"]["selector_vs_union"][
        "positive_incremental_year_count"
    ] = 3
    failing = apply_development_gates(metrics)
    assert failing["passed"] is False
    assert "stress_at_least_four_positive_incremental_years" in failing["failures"]
