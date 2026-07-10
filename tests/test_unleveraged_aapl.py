from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.deterministic_aapl import CostAssumptions, EvaluationPeriod
from agent_benchmark.unleveraged_aapl import (
    CONTEXTUAL_EXHAUSTION_V1,
    GAP_DOWN_CASH_V1,
    LongCashSpec,
    assert_unleveraged_ledger,
    build_long_cash_target,
    canonical_context_frame,
    evaluate_continuous_account,
    evaluate_fresh_periods,
    simulate_unleveraged_period,
    run_unleveraged_experiment,
)


def context_frame(periods: int = 320, start: str = "2022-01-03") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods)
    aapl = np.linspace(100.0, 160.0, periods)
    spy = np.linspace(300.0, 390.0, periods)
    qqq = np.linspace(250.0, 370.0, periods)
    return pd.DataFrame(
        {
            "aapl_open": aapl * 0.999,
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": spy,
            "qqq_adj_close": qqq,
        },
        index=index,
    )


def test_spec_rejects_selection_cutoff_that_touches_final_period():
    spec = LongCashSpec(
        name="bad",
        rule_type="gap_down",
        selection_data_cutoff="2024-01-01",
    )
    with pytest.raises(ValueError, match="before 2024"):
        spec.validate()


def test_contextual_rule_uses_prior_percentile_and_next_open_shift():
    frame = context_frame(140)
    # Force the decision day's completed intraday return above every prior value.
    decision_index = 130
    frame.iloc[decision_index, frame.columns.get_loc("aapl_open")] *= 0.8
    # Both market filters are negative on that same completed close.
    frame.iloc[decision_index - 10 : decision_index + 1, frame.columns.get_loc("spy_adj_close")] = np.linspace(350, 300, 11)
    frame.iloc[decision_index - 10 : decision_index + 1, frame.columns.get_loc("qqq_adj_close")] = np.linspace(350, 300, 11)
    spec = LongCashSpec(
        name="toy",
        rule_type="contextual_exhaustion",
        aapl_percentile_lookback=126,
        aapl_percentile=0.90,
        market_return_lookback=10,
        cash_sessions=1,
    )
    target = build_long_cash_target(frame, spec)
    assert target.iloc[decision_index] == 0.0
    period = EvaluationPeriod(
        "toy",
        frame.index[decision_index].date().isoformat(),
        frame.index[decision_index + 2].date().isoformat(),
    )
    ledger = simulate_unleveraged_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
    )
    # Fill on the decision day still executes yesterday's target; cash starts next open.
    assert ledger.loc[0, "target_exposure"] == 1.0
    assert ledger.loc[1, "decision_date"] == frame.index[decision_index].date().isoformat()
    assert ledger.loc[1, "target_exposure"] == 0.0


def test_no_leverage_wrapper_rejects_margin_configuration():
    frame = context_frame()
    target = pd.Series(1.0, index=frame.index)
    period = EvaluationPeriod("toy", "2023-01-03", "2023-02-01")
    with pytest.raises(ValueError, match="annual_margin_rate=0"):
        simulate_unleveraged_period(
            frame,
            target,
            period,
            CostAssumptions(annual_margin_rate=0.08),
        )


def test_no_leverage_wrapper_rejects_fractional_target():
    frame = context_frame()
    target = pd.Series(1.0, index=frame.index)
    target.iloc[260] = 0.5
    period = EvaluationPeriod("toy", frame.index[250].date().isoformat(), frame.index[-1].date().isoformat())
    with pytest.raises(ValueError, match="binary LONG/CASH"):
        simulate_unleveraged_period(
            frame,
            target,
            period,
            CostAssumptions(annual_margin_rate=0.0),
        )


def test_long_cash_execution_has_no_borrowing_shorting_or_exposure_above_one():
    frame = context_frame()
    # Synthetic gap creates at least one cash round trip.
    frame.iloc[260, frame.columns.get_loc("aapl_open")] *= 0.90
    target = build_long_cash_target(frame, GAP_DOWN_CASH_V1)
    period = EvaluationPeriod(
        "toy",
        frame.index[250].date().isoformat(),
        frame.index[-1].date().isoformat(),
    )
    ledger = simulate_unleveraged_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=20.0, annual_margin_rate=0.0),
    )
    proof = assert_unleveraged_ledger(ledger)
    assert proof["passed"] is True
    assert proof["minimum_cash"] >= -1e-9
    assert proof["minimum_shares"] >= 0.0
    assert proof["maximum_requested_target"] <= 1.0
    assert proof["maximum_post_fill_exposure"] <= 1.0 + 1e-9
    assert proof["maximum_holding_exposure"] <= 1.0 + 1e-9
    assert proof["total_margin_interest"] == 0.0
    assert set(ledger["target_exposure"].unique()) <= {0.0, 1.0}


def test_invariant_audit_rejects_negative_cash_even_with_safe_target():
    frame = context_frame()
    target = pd.Series(1.0, index=frame.index)
    period = EvaluationPeriod("toy", "2023-01-03", "2023-02-01")
    ledger = simulate_unleveraged_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
    )
    ledger.loc[0, "cash"] = -0.01
    with pytest.raises(RuntimeError, match="negative_cash"):
        assert_unleveraged_ledger(ledger)


@pytest.mark.parametrize("corrupt", [np.nan, np.inf, -np.inf])
def test_invariant_audit_rejects_nonfinite_values(corrupt):
    frame = context_frame()
    target = pd.Series(1.0, index=frame.index)
    period = EvaluationPeriod("toy", "2023-01-03", "2023-02-01")
    ledger = simulate_unleveraged_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
    )
    ledger.loc[0, "new_exposure_after_fill"] = corrupt
    with pytest.raises(RuntimeError, match="nonfinite_values"):
        assert_unleveraged_ledger(ledger)


def test_identical_long_cash_target_cannot_false_pass_against_buy_hold():
    frame = context_frame()
    spec = LongCashSpec(name="never_gap", rule_type="gap_down", gap_threshold=-0.49)
    period = EvaluationPeriod(
        "toy", frame.index[250].date().isoformat(), frame.index[-1].date().isoformat()
    )
    report, _ = evaluate_fresh_periods(
        frame,
        spec,
        [period],
        CostAssumptions(slippage_bps=5.0, annual_margin_rate=0.0),
    )
    item = report["periods"]["toy"]
    assert item["excess_return_vs_aapl_buy_hold"] == 0.0
    assert item["requested_success"] is False
    assert report["all_periods_requested_success"] is False


def test_continuous_account_does_not_reset_at_year_boundary():
    frame = context_frame(520, start="2022-01-03")
    periods = (
        EvaluationPeriod("2022", "2022-01-05", "2022-12-30"),
        EvaluationPeriod("2023", "2023-01-03", "2023-12-29"),
    )
    spec = LongCashSpec(name="never_gap", rule_type="gap_down", gap_threshold=-0.49)
    report, ledger = evaluate_continuous_account(
        frame,
        spec,
        periods,
        CostAssumptions(slippage_bps=0.0, annual_margin_rate=0.0),
    )
    assert len(ledger) > 500
    assert ledger["strategy_trade_executed"].sum() == 1
    assert report["full_span"]["excess_return_vs_aapl_buy_hold"] == 0.0
    assert all(value == pytest.approx(0.0) for value in report["period_active_log_returns"].values())


def test_default_contextual_spec_is_binary_and_frozen_before_2024():
    frame = context_frame()
    target = build_long_cash_target(frame, CONTEXTUAL_EXHAUSTION_V1).dropna()
    assert set(target.unique()) <= {0.0, 1.0}
    assert CONTEXTUAL_EXHAUSTION_V1.selection_data_cutoff == "2023-12-31"


def test_promotion_runner_rejects_noncanonical_periods_before_io(tmp_path):
    with pytest.raises(ValueError, match="exact ordered"):
        run_unleveraged_experiment(
            repo_root=tmp_path,
            output_dir=tmp_path / "runs",
            cache_path=tmp_path / "cache.csv",
            spec=GAP_DOWN_CASH_V1,
            periods=[EvaluationPeriod("only_2025", "2025-01-01", "2025-12-31")],
        )


@pytest.mark.parametrize("touch_count", [0, -1, True, 1.5])
def test_promotion_runner_rejects_invalid_holdout_touch_count(tmp_path, touch_count):
    with pytest.raises(ValueError, match="at least one"):
        run_unleveraged_experiment(
            repo_root=tmp_path,
            output_dir=tmp_path / "runs",
            cache_path=tmp_path / "cache.csv",
            spec=GAP_DOWN_CASH_V1,
            final_holdout_touch_count=touch_count,
        )
