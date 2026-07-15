from __future__ import annotations

import pandas as pd

from agent_benchmark import contextual_expert_aggregation_2024_audit_evidence as evidence
from agent_benchmark import contextual_expert_aggregation_ledger as ledger


def _series(values: list[float | int]) -> pd.Series:
    index = pd.bdate_range("2005-01-03", periods=len(values))
    return pd.Series(values, index=index)


def test_verified_continuation_preserves_parent_hash_chain_and_no_leverage() -> None:
    opens = _series([10.0, 10.5, 10.2, 10.8, 11.0, 10.7])
    targets = _series([1, 1, 0, 0, 1, 1])
    parent = ledger.run_continuous_ledger(
        opens.iloc[:3],
        targets.iloc[:3],
        policy_name="online_full",
        cost_bps=5.0,
    )
    result = evidence.continue_verified_account(
        prior_ledger=parent.ledger,
        parent_state=parent.state,
        suffix_adjusted_opens=opens.iloc[3:],
        suffix_close_targets=targets.iloc[3:],
    )
    assert len(result.full_ledger) == 6
    assert result.suffix_ledger.iloc[0]["previous_row_sha256"] == parent.state.ledger_tip_sha256
    assert result.no_leverage_proof["maximum_gross_exposure"] == 1


def test_fill_xor_includes_decision_dates_and_reconciles_open_or_complete_runs() -> None:
    opens = _series([10.0, 10.5, 10.2, 10.8, 11.0, 10.7, 10.9, 11.2])
    lead_targets = _series([1, 1, 1, 1, 1, 1, 1, 1])
    shadow_targets = _series([1, 1, 1, 0, 0, 1, 1, 1])
    parent_targets = lead_targets.iloc[:3]
    lead_parent = ledger.run_continuous_ledger(
        opens.iloc[:3], parent_targets, policy_name="online_full", cost_bps=10.0
    )
    shadow_parent = ledger.run_continuous_ledger(
        opens.iloc[:3], parent_targets, policy_name="online_full", cost_bps=10.0
    )
    lead = evidence.continue_verified_account(
        prior_ledger=lead_parent.ledger,
        parent_state=lead_parent.state,
        suffix_adjusted_opens=opens.iloc[3:],
        suffix_close_targets=lead_targets.iloc[3:],
    )
    shadow = evidence.continue_verified_account(
        prior_ledger=shadow_parent.ledger,
        parent_state=shadow_parent.state,
        suffix_adjusted_opens=opens.iloc[3:],
        suffix_close_targets=shadow_targets.iloc[3:],
    )
    xor = evidence.extract_audit_xor_evidence(
        full_shadow=shadow.full_ledger,
        full_lead=lead.full_ledger,
        shadow_parent_state=shadow_parent.state,
        lead_parent_state=lead_parent.state,
        shadow_terminal_state=shadow.terminal_state,
        lead_terminal_state=lead.terminal_state,
    )
    assert xor.reconciliation["passed"] is True
    assert xor.reconciliation["economic_xor_fill_count"] > 0
    assert len(xor.complete) == 1
    assert xor.complete.iloc[0]["orientation"] == evidence.SHADOW_CASH_LEAD_LONG
    assert xor.complete.iloc[0]["start_shadow_decision_date"]
    assert xor.complete.iloc[0]["end_lead_decision_date"]


def test_cash_episode_evidence_keeps_terminal_open_episode_unclosed() -> None:
    opens = _series([10.0, 10.5, 10.2, 10.8, 11.0])
    targets = _series([1, 1, 0, 0, 0])
    run = ledger.run_continuous_ledger(
        opens, targets, policy_name="online_full", cost_bps=5.0
    )
    episodes = evidence.cash_episode_evidence(
        ledger=run.ledger,
        terminal_state=run.state,
    )
    assert episodes.reconciliation["passed"] is True
    assert episodes.reconciliation["terminal_episode_force_closed"] is False
    assert len(episodes.open) == 1
