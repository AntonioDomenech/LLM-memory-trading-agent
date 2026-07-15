from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_ledger as ledger


Error = ledger.BinaryLedgerError


def _series(
    dates: list[str], values: list[float | int], *, name: str
) -> pd.Series:
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=name)


def _run(
    dates: list[str],
    opens: list[float],
    targets: list[int | float],
    *,
    policy: str = "online",
    cost_bps: float = 5.0,
) -> ledger.LedgerRun:
    return ledger.run_continuous_ledger(
        _series(dates, opens, name="aapl_adj_open"),
        _series(dates, targets, name="target"),
        policy_name=policy,
        cost_bps=cost_bps,
    )


def _benchmark(
    dates: list[str], opens: list[float], *, cost_bps: float = 5.0
) -> ledger.LedgerRun:
    return _run(
        dates,
        opens,
        [1] * len(dates),
        policy="aapl_buy_hold",
        cost_bps=cost_bps,
    )


def _rehash_row(frame: pd.DataFrame, row_index: int) -> None:
    row = frame.to_dict(orient="records")[row_index]
    frame.at[frame.index[row_index], "row_sha256"] = ledger._sha256(
        {name: row[name] for name in ledger.LEDGER_COLUMNS[:-1]}
    )


def test_common_inception_is_forced_long_once_and_decisions_fill_next_open() -> None:
    dates = ["2004-12-31", "2005-01-03", "2005-01-04", "2005-01-05"]
    result = _run(dates, [90.0, 100.0, 110.0, 105.0], [0, 0, 1, 1])
    rows = result.ledger
    cost = 5.0 / 10_000.0

    assert rows["fill_date"].tolist() == dates[1:]
    assert rows.iloc[0]["decision_date"] == ""
    assert rows.iloc[0]["transition"] == "INCEPTION"
    assert rows.iloc[0]["requested_target_exposure"] == 1
    assert rows.iloc[0]["shares_after_fill"] == pytest.approx(
        1000.0 / (100.0 * (1.0 + cost)), abs=1e-15
    )
    assert rows.iloc[0]["cash_after_fill"] == 0.0
    assert rows.iloc[1]["decision_date"] == "2005-01-03"
    assert rows.iloc[1]["transition"] == "SELL"
    assert rows.iloc[1]["requested_target_exposure"] == 0
    assert rows.iloc[2]["decision_date"] == "2005-01-04"
    assert rows.iloc[2]["transition"] == "BUY"
    assert result.state.inception_count == 1
    assert result.state.pending_decision_date == "2005-01-05"
    assert result.state.pending_target_exposure == 1
    assert ledger.verify_ledger(
        rows,
        start_state=ledger.AccountState.initial(
            policy_name="online", cost_bps=5.0
        ),
        expected_end_state=result.state,
    ) == result.state


def test_full_run_equals_split_continuation_with_cutoff_close_target_pending() -> None:
    dates = [
        "2005-01-03",
        "2005-01-04",
        "2005-01-05",
        "2018-12-31",
        "2019-01-02",
        "2019-01-03",
    ]
    opens = [100.0, 102.0, 101.0, 99.0, 104.0, 103.0]
    targets = [1, 0, 1, 0, 1, 1]
    full = _run(dates, opens, targets)
    development = _run(dates[:4], opens[:4], targets[:4])

    assert development.state.last_session_date == "2018-12-31"
    assert development.state.pending_decision_date == "2018-12-31"
    assert development.state.pending_target_exposure == 0
    checkpoint = development.state.to_checkpoint()
    verified_state = ledger.verify_continuation_checkpoint(
        _series(dates[:4], opens[:4], name="aapl_adj_open"),
        _series(dates[:4], targets[:4], name="target"),
        development.ledger,
        checkpoint,
        policy_name="online",
        cost_bps=5.0,
    )
    assert verified_state == development.state
    confirmation = ledger.run_verified_continuation(
        _series(dates[:4], opens[:4], name="aapl_adj_open"),
        _series(dates[:4], targets[:4], name="target"),
        development.ledger,
        checkpoint,
        _series(dates[4:], opens[4:], name="aapl_adj_open"),
        _series(dates[4:], targets[4:], name="target"),
        policy_name="online",
        cost_bps=5.0,
    )
    combined = pd.concat(
        [development.ledger, confirmation.ledger], ignore_index=True
    )

    pd.testing.assert_frame_equal(combined, full.ledger)
    assert confirmation.state == full.state
    assert ledger.AccountState.from_checkpoint(checkpoint) == development.state


def test_continuation_rejects_unproved_and_self_consistently_tampered_state() -> None:
    prefix_dates = ["2005-01-03", "2005-01-04", "2018-12-31"]
    prefix_opens = [100.0, 102.0, 101.0]
    prefix_targets = [1, 0, 0]
    prefix = _run(prefix_dates, prefix_opens, prefix_targets)

    assert not hasattr(ledger, "VerifiedContinuation")

    changed_prior_ledger = prefix.ledger.copy()
    changed_prior_ledger.loc[1, "fill_price"] += 0.01
    with pytest.raises(Error, match="Prior ledger differs"):
        ledger.verify_continuation_checkpoint(
            _series(prefix_dates, prefix_opens, name="aapl_adj_open"),
            _series(prefix_dates, prefix_targets, name="target"),
            changed_prior_ledger,
            prefix.state.to_checkpoint(),
            policy_name="online",
            cost_bps=5.0,
        )
    boolean_prior_ledger = prefix.ledger.copy()
    boolean_prior_ledger["requested_target_exposure"] = boolean_prior_ledger[
        "requested_target_exposure"
    ].astype(object)
    boolean_prior_ledger.at[0, "requested_target_exposure"] = True
    with pytest.raises(Error, match="Prior ledger differs"):
        ledger.verify_continuation_checkpoint(
            _series(prefix_dates, prefix_opens, name="aapl_adj_open"),
            _series(prefix_dates, prefix_targets, name="target"),
            boolean_prior_ledger,
            prefix.state.to_checkpoint(),
            policy_name="online",
            cost_bps=5.0,
        )

    tampered_state = replace(
        prefix.state,
        cumulative_active_log_edge=prefix.state.cumulative_active_log_edge + 0.1,
    )
    tampered_checkpoint = tampered_state.to_checkpoint()
    assert (
        ledger.AccountState.from_checkpoint(tampered_checkpoint)
        == tampered_state
    )
    with pytest.raises(Error, match="differs from full prefix replay"):
        ledger.verify_continuation_checkpoint(
            _series(prefix_dates, prefix_opens, name="aapl_adj_open"),
            _series(prefix_dates, prefix_targets, name="target"),
            prefix.ledger,
            tampered_checkpoint,
            policy_name="online",
            cost_bps=5.0,
        )

    with pytest.raises(Error, match="differs from full prefix replay"):
        ledger.run_verified_continuation(
            _series(prefix_dates, prefix_opens, name="aapl_adj_open"),
            _series(prefix_dates, prefix_targets, name="target"),
            prefix.ledger,
            tampered_checkpoint,
            object(),
            object(),
            policy_name="online",
            cost_bps=5.0,
        )


@pytest.mark.parametrize("cost_bps", [5.0, 10.0])
def test_cost_is_charged_exactly_on_both_changing_legs(cost_bps: float) -> None:
    dates = [
        "2005-01-03",
        "2005-01-04",
        "2005-01-05",
        "2005-01-06",
        "2005-01-07",
    ]
    opens = [100.0, 105.0, 120.0, 90.0, 80.0]
    result = _run(dates, opens, [1, 0, 0, 1, 1], cost_bps=cost_bps)
    sell = result.ledger.loc[result.ledger["transition"].eq("SELL")].iloc[0]
    buy = result.ledger.loc[result.ledger["transition"].eq("BUY")].iloc[0]
    c = cost_bps / 10_000.0

    assert sell["fill_price"] == 120.0 * (1.0 - c)
    assert sell["shares_after_fill"] == 0.0
    assert sell["cash_after_fill"] > 0.0
    assert buy["fill_price"] == 80.0 * (1.0 + c)
    assert buy["cash_after_fill"] == 0.0
    assert buy["shares_after_fill"] == pytest.approx(
        sell["cash_after_fill"] / buy["fill_price"], abs=1e-15
    )
    episodes = ledger.extract_cash_episodes(
        result.ledger,
        start_state=ledger.AccountState.initial(
            policy_name="online", cost_bps=cost_bps
        ),
        expected_end_state=result.state,
    )
    expected = math.log(120.0 / 80.0) + math.log((1.0 - c) / (1.0 + c))
    assert len(episodes.complete) == 1
    assert episodes.complete.iloc[0]["net_active_log_edge"] == pytest.approx(
        expected, abs=1e-15
    )
    assert episodes.complete.iloc[0]["cash_fill_observations"] == 2


def test_binary_alternation_rebalances_only_when_target_changes() -> None:
    dates = pd.bdate_range("2005-01-03", periods=7).strftime("%Y-%m-%d").tolist()
    targets = [1, 0, 0, 1, 1, 0, 1]
    result = _run(dates, [100.0 + i for i in range(7)], targets)
    transitions = result.ledger["transition"].tolist()

    assert transitions == [
        "INCEPTION",
        "HOLD_LONG",
        "SELL",
        "HOLD_CASH",
        "BUY",
        "HOLD_LONG",
        "SELL",
    ]
    holds = result.ledger["transition"].str.startswith("HOLD")
    assert (result.ledger.loc[holds, "signed_share_delta"] == 0.0).all()
    assert (~result.ledger.loc[holds, "trade_executed"]).all()
    assert result.state.pending_target_exposure == 1


def test_final_close_target_remains_pending_without_terminal_sale() -> None:
    dates = ["2005-01-03", "2005-01-04", "2005-01-05"]
    result = _run(dates, [100.0, 110.0, 120.0], [1, 1, 0])
    episodes = ledger.extract_cash_episodes(
        result.ledger,
        start_state=ledger.AccountState.initial(
            policy_name="online", cost_bps=5.0
        ),
    )

    assert result.state.held_target == 1
    assert result.state.pending_target_exposure == 0
    assert "SELL" not in result.ledger["transition"].tolist()
    assert episodes.complete.empty
    assert episodes.unresolved.iloc[0]["status"] == "pending_cash_entry_unexecuted"
    assert episodes.unresolved.iloc[0]["net_active_log_edge_to_mark"] == 0.0


def test_open_terminal_cash_episode_is_reported_and_complete_reconciliation_rejects() -> None:
    dates = ["2005-01-03", "2005-01-04", "2005-01-05"]
    opens = [100.0, 110.0, 90.0]
    strategy = _run(dates, opens, [0, 0, 0])
    benchmark = _benchmark(dates, opens)
    extraction = ledger.extract_cash_episodes(
        strategy.ledger,
        start_state=ledger.AccountState.initial(
            policy_name="online", cost_bps=5.0
        ),
    )

    assert extraction.complete.empty
    assert extraction.unresolved.iloc[0]["status"] == "open_cash"
    with pytest.raises(Error, match="terminal CASH episode"):
        ledger.reconcile_complete_cash_episodes(
            strategy.ledger,
            benchmark.ledger,
            extraction,
            strategy_start_state=ledger.AccountState.initial(
                policy_name="online", cost_bps=5.0
            ),
            buy_hold_start_state=ledger.AccountState.initial(
                policy_name="aapl_buy_hold", cost_bps=5.0
            ),
        )


def test_checkpoint_schema_hash_and_numeric_state_are_fail_closed() -> None:
    result = _run(
        ["2005-01-03", "2005-01-04"], [100.0, 101.0], [1, 1]
    )
    checkpoint = result.state.to_checkpoint()
    assert ledger.AccountState.from_checkpoint(checkpoint) == result.state

    extra = {**checkpoint, "unexpected": True}
    with pytest.raises(Error, match="exact frozen schema"):
        ledger.AccountState.from_checkpoint(extra)
    tampered = json.loads(json.dumps(checkpoint))
    tampered["account_state"]["cash"] = 1.0
    with pytest.raises(Error, match="self-hash"):
        ledger.AccountState.from_checkpoint(tampered)
    boolean_checkpoint_schema = json.loads(json.dumps(checkpoint))
    boolean_checkpoint_schema["checkpoint_schema_version"] = True
    boolean_checkpoint_schema["account_state_sha256"] = ledger._sha256(
        {
            "checkpoint_schema_version": True,
            "account_state": boolean_checkpoint_schema["account_state"],
        }
    )
    with pytest.raises(Error, match="schema version"):
        ledger.AccountState.from_checkpoint(boolean_checkpoint_schema)
    boolean_state_schema = json.loads(json.dumps(checkpoint))
    boolean_state_schema["account_state"]["schema_version"] = True
    boolean_state_schema["account_state_sha256"] = ledger._sha256(
        {
            "checkpoint_schema_version": ledger.CHECKPOINT_SCHEMA_VERSION,
            "account_state": boolean_state_schema["account_state"],
        }
    )
    with pytest.raises(Error, match="schema version"):
        ledger.AccountState.from_checkpoint(boolean_state_schema)
    for bad in (-1.0, math.nan, math.inf, sys.float_info.min / 2.0):
        with pytest.raises(Error):
            replace(result.state, last_equity=bad)
    with pytest.raises(Error, match="canonical integer"):
        replace(result.state, pending_target_exposure=0.5)
    with pytest.raises(Error, match="canonical integer"):
        replace(result.state, pending_target_exposure=True)
    with pytest.raises(Error, match="canonical float"):
        replace(result.state, cost_bps=5)


def test_exact_genesis_session_is_mandatory_and_late_reset_is_rejected() -> None:
    with pytest.raises(Error, match="omits the exact inception"):
        _run(["2019-01-02", "2019-01-03"], [100.0, 101.0], [1, 1])
    with pytest.raises(Error, match="omits the exact inception"):
        _run(["2005-01-04"], [100.0], [1])


@pytest.mark.parametrize("target", [-1.0, 0.5, math.nan, math.inf, sys.float_info.min / 2.0])
def test_targets_reject_nonbinary_nonfinite_and_subnormal(target: float) -> None:
    with pytest.raises(Error):
        _run(["2005-01-03"], [100.0], [target])


@pytest.mark.parametrize("price", [-1.0, 0.0, math.nan, math.inf, sys.float_info.min / 2.0])
def test_prices_reject_negative_zero_nonfinite_and_subnormal(price: float) -> None:
    with pytest.raises(Error):
        _run(["2005-01-03"], [price], [1])


def test_no_leverage_proof_and_row_replay_reject_tamper() -> None:
    result = _run(
        ["2005-01-03", "2005-01-04", "2005-01-05"],
        [100.0, 101.0, 102.0],
        [1, 0, 1],
    )
    proof = ledger.assert_no_leverage(result.ledger)
    assert proof["passed"] is True
    assert proof["maximum_gross_exposure"] == 1

    negative = result.ledger.copy()
    negative.loc[0, "cash_after_fill"] = -1.0
    with pytest.raises(Error):
        ledger.assert_no_leverage(negative)
    negative_before = result.ledger.copy()
    negative_before.loc[1, "cash_before_fill"] = -1.0
    with pytest.raises(Error):
        ledger.assert_no_leverage(negative_before)
    boolean_row = result.ledger.copy()
    boolean_row["requested_target_exposure"] = boolean_row[
        "requested_target_exposure"
    ].astype(object)
    boolean_row.at[0, "requested_target_exposure"] = True
    with pytest.raises(Error, match="canonical integer"):
        ledger.verify_ledger(
            boolean_row,
            start_state=ledger.AccountState.initial(
                policy_name="online", cost_bps=5.0
            ),
        )
    changed = result.ledger.copy()
    changed.loc[1, "fill_price"] += 1.0
    with pytest.raises(Error, match="self-hash"):
        ledger.verify_ledger(
            changed,
            start_state=ledger.AccountState.initial(
                policy_name="online", cost_bps=5.0
            ),
        )


def test_cross_cost_actions_are_exact_and_tamper_is_rejected() -> None:
    dates = pd.bdate_range("2005-01-03", periods=6).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 102.0, 99.0, 98.0, 105.0, 107.0]
    targets = [1, 0, 0, 1, 1, 0]
    base = _run(dates, opens, targets, cost_bps=5.0)
    stress = _run(dates, opens, targets, cost_bps=10.0)

    proof = ledger.assert_cross_cost_action_identity(
        {"base_5bps": base.ledger, "stress_10bps": stress.ledger}
    )
    assert proof["passed"] is True
    assert proof["cost_scenario_count"] == 2
    assert proof["observed_cost_bps"] == [5.0, 10.0]
    assert proof["shared_input_stream_sha256"].startswith("sha256:")
    altered = stress.ledger.copy()
    altered.loc[2, "close_decision_target_exposure"] = 1
    with pytest.raises(Error):
        ledger.assert_cross_cost_action_identity(
            {"base_5bps": base.ledger, "stress_10bps": altered}
        )


def test_cross_cost_proof_rejects_duplicate_ledger_under_different_labels() -> None:
    dates = pd.bdate_range("2005-01-03", periods=3).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 101.0, 102.0]
    base = _run(dates, opens, [1, 0, 1], cost_bps=5.0).ledger
    with pytest.raises(Error, match="distinct cost_bps"):
        ledger.assert_cross_cost_action_identity(
            {"claimed_base": base, "claimed_stress": base.copy()}
        )


def test_cross_cost_proof_rejects_different_exogenous_price_paths() -> None:
    dates = pd.bdate_range("2005-01-03", periods=4).strftime("%Y-%m-%d").tolist()
    targets = [1, 0, 1, 1]
    base = _run(dates, [100.0, 101.0, 99.0, 102.0], targets, cost_bps=5.0)
    stress = _run(
        dates,
        [100.0, 101.0, 98.5, 102.0],
        targets,
        cost_bps=10.0,
    )
    with pytest.raises(Error, match="Exogenous input stream"):
        ledger.assert_cross_cost_action_identity(
            {"base_5bps": base.ledger, "stress_10bps": stress.ledger}
        )


def test_cross_cost_proof_replays_self_rehashed_semantically_impossible_rows() -> None:
    dates = ["2005-01-03"]
    opens = [100.0]
    base = _run(dates, opens, [1], cost_bps=5.0).ledger.copy()
    stress = _run(dates, opens, [1], cost_bps=10.0).ledger.copy()
    for frame in (base, stress):
        frame.at[frame.index[0], "transition"] = "HOLD_LONG"
        _rehash_row(frame, 0)
    with pytest.raises(Error, match="deterministic replay"):
        ledger.assert_cross_cost_action_identity(
            {"base_5bps": base, "stress_10bps": stress}
        )


@pytest.mark.parametrize("cost_bps", [5.0, 10.0])
def test_always_long_is_exactly_same_ledger_buy_hold(cost_bps: float) -> None:
    dates = pd.bdate_range("2005-01-03", periods=5).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 98.0, 105.0, 101.0, 110.0]
    always = _run(
        dates, opens, [1] * len(dates), policy="always_long", cost_bps=cost_bps
    )
    benchmark = _benchmark(dates, opens, cost_bps=cost_bps)

    assert ledger.assert_always_long_matches_buy_hold(
        always.ledger, benchmark.ledger
    )["passed"] is True
    assert always.ledger["cumulative_active_log_edge"].iloc[-1] == 0.0


def test_always_long_rejects_an_unexecuted_final_cash_target() -> None:
    dates = pd.bdate_range("2005-01-03", periods=4).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 101.0, 102.0, 103.0]
    always = _run(dates, opens, [1, 1, 1, 0], policy="always_long")
    benchmark = _run(dates, opens, [1, 1, 1, 0], policy="aapl_buy_hold")
    with pytest.raises(Error, match="not always LONG"):
        ledger.assert_always_long_matches_buy_hold(always.ledger, benchmark.ledger)


def test_complete_multi_session_cash_episodes_reconcile_to_ledger_edge() -> None:
    dates = pd.bdate_range("2005-01-03", periods=8).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 105.0, 120.0, 110.0, 90.0, 80.0, 100.0, 102.0]
    targets = [1, 0, 0, 0, 1, 1, 1, 1]
    strategy = _run(dates, opens, targets)
    benchmark = _benchmark(dates, opens)
    start = ledger.AccountState.initial(policy_name="online", cost_bps=5.0)
    extraction = ledger.extract_cash_episodes(strategy.ledger, start_state=start)
    proof = ledger.reconcile_complete_cash_episodes(
        strategy.ledger,
        benchmark.ledger,
        extraction,
        strategy_start_state=start,
        buy_hold_start_state=ledger.AccountState.initial(
            policy_name="aapl_buy_hold", cost_bps=5.0
        ),
    )

    assert len(extraction.complete) == 1
    assert extraction.complete.iloc[0]["cash_fill_observations"] == 3
    assert extraction.unresolved.empty
    assert abs(proof["reconciliation_error"]) <= 1e-10


def test_episode_reconciliation_rejects_valid_non_buy_hold_comparator() -> None:
    dates = pd.bdate_range("2005-01-03", periods=5).strftime("%Y-%m-%d").tolist()
    opens = [100.0] * len(dates)
    strategy = _run(
        dates,
        opens,
        [1] * len(dates),
        policy="strategy",
        cost_bps=0.0,
    )
    cash_round_trip = _run(
        dates,
        opens,
        [1, 0, 1, 1, 1],
        policy="decoy_comparator",
        cost_bps=0.0,
    )
    strategy_start = ledger.AccountState.initial(
        policy_name="strategy", cost_bps=0.0
    )
    extraction = ledger.extract_cash_episodes(
        strategy.ledger, start_state=strategy_start
    )
    assert extraction.complete.empty
    assert cash_round_trip.state.held_target == 1
    with pytest.raises(Error, match="buy-and-hold.*CASH"):
        ledger.reconcile_complete_cash_episodes(
            strategy.ledger,
            cash_round_trip.ledger,
            extraction,
            strategy_start_state=strategy_start,
            buy_hold_start_state=ledger.AccountState.initial(
                policy_name="decoy_comparator", cost_bps=0.0
            ),
        )


def test_cash_episode_identity_is_cost_independent_and_economics_are_not() -> None:
    dates = pd.bdate_range("2005-01-03", periods=6).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 105.0, 120.0, 90.0, 80.0, 100.0]
    targets = [1, 0, 0, 1, 1, 1]
    base = _run(dates, opens, targets, cost_bps=5.0)
    stress = _run(dates, opens, targets, cost_bps=10.0)
    base_episode = ledger.extract_cash_episodes(
        base.ledger,
        start_state=ledger.AccountState.initial(
            policy_name="online", cost_bps=5.0
        ),
    ).complete.iloc[0]
    stress_episode = ledger.extract_cash_episodes(
        stress.ledger,
        start_state=ledger.AccountState.initial(
            policy_name="online", cost_bps=10.0
        ),
    ).complete.iloc[0]
    assert base_episode["episode_id"] == stress_episode["episode_id"]
    assert base_episode["entry_fill_date"] == stress_episode["entry_fill_date"]
    assert base_episode["exit_fill_date"] == stress_episode["exit_fill_date"]
    assert base_episode["episode_sha256"] != stress_episode["episode_sha256"]
    assert base_episode["net_active_log_edge"] > stress_episode["net_active_log_edge"]


def test_cash_episode_reconciliation_rejects_tampered_rows_hashes_and_schema() -> None:
    dates = pd.bdate_range("2005-01-03", periods=6).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 105.0, 120.0, 90.0, 80.0, 100.0]
    strategy = _run(dates, opens, [1, 0, 0, 1, 1, 1])
    benchmark = _benchmark(dates, opens)
    strategy_start = ledger.AccountState.initial(
        policy_name="online", cost_bps=5.0
    )
    benchmark_start = ledger.AccountState.initial(
        policy_name="aapl_buy_hold", cost_bps=5.0
    )
    extracted = ledger.extract_cash_episodes(
        strategy.ledger, start_state=strategy_start
    )
    tampered_complete = extracted.complete.copy()
    tampered_complete.at[0, "entry_fill_date"] = "2099-01-01"
    tampered_complete.at[0, "episode_sha256"] = "sha256:" + "0" * 64
    tampered = ledger.EpisodeExtraction(
        complete=tampered_complete,
        unresolved=extracted.unresolved.copy(),
    )
    with pytest.raises(Error, match="deterministic regeneration"):
        ledger.reconcile_complete_cash_episodes(
            strategy.ledger,
            benchmark.ledger,
            tampered,
            strategy_start_state=strategy_start,
            buy_hold_start_state=benchmark_start,
        )
    extra_column = ledger.EpisodeExtraction(
        complete=extracted.complete.assign(extra=1),
        unresolved=extracted.unresolved.copy(),
    )
    with pytest.raises(Error, match="exact canonical columns"):
        ledger.reconcile_complete_cash_episodes(
            strategy.ledger,
            benchmark.ledger,
            extra_column,
            strategy_start_state=strategy_start,
            buy_hold_start_state=benchmark_start,
        )


def test_signed_xor_primary_cash_and_reverse_orientation_reconcile() -> None:
    dates = pd.bdate_range("2005-01-03", periods=6).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 105.0, 120.0, 90.0, 80.0, 100.0]
    cash_policy = _run(dates, opens, [1, 0, 0, 1, 1, 1], policy="primary")
    long_policy = _run(dates, opens, [1] * 6, policy="comparator")
    primary_start = ledger.AccountState.initial(
        policy_name="primary", cost_bps=5.0
    )
    comparator_start = ledger.AccountState.initial(
        policy_name="comparator", cost_bps=5.0
    )
    forward = ledger.extract_signed_xor_episodes(
        cash_policy.ledger,
        long_policy.ledger,
        primary_start_state=primary_start,
        comparator_start_state=comparator_start,
    )
    forward_proof = ledger.reconcile_signed_xor(
        cash_policy.ledger,
        long_policy.ledger,
        forward,
        primary_start_state=primary_start,
        comparator_start_state=comparator_start,
    )

    assert (
        forward.complete.iloc[0]["orientation"]
        == "primary_cash_comparator_long"
    )
    assert abs(forward_proof["reconciliation_error"]) <= 1e-10

    reverse = ledger.extract_signed_xor_episodes(
        long_policy.ledger,
        cash_policy.ledger,
        primary_start_state=comparator_start,
        comparator_start_state=primary_start,
    )
    reverse_proof = ledger.reconcile_signed_xor(
        long_policy.ledger,
        cash_policy.ledger,
        reverse,
        primary_start_state=comparator_start,
        comparator_start_state=primary_start,
    )
    assert (
        reverse.complete.iloc[0]["orientation"]
        == "primary_long_comparator_cash"
    )
    assert reverse.complete.iloc[0]["net_signed_log_edge"] == pytest.approx(
        -forward.complete.iloc[0]["net_signed_log_edge"], abs=1e-15
    )
    assert abs(reverse_proof["reconciliation_error"]) <= 1e-10

    cash_policy_stress = _run(
        dates,
        opens,
        [1, 0, 0, 1, 1, 1],
        policy="primary",
        cost_bps=10.0,
    )
    long_policy_stress = _run(
        dates,
        opens,
        [1] * 6,
        policy="comparator",
        cost_bps=10.0,
    )
    stress = ledger.extract_signed_xor_episodes(
        cash_policy_stress.ledger,
        long_policy_stress.ledger,
        primary_start_state=ledger.AccountState.initial(
            policy_name="primary", cost_bps=10.0
        ),
        comparator_start_state=ledger.AccountState.initial(
            policy_name="comparator", cost_bps=10.0
        ),
    )
    assert forward.complete.iloc[0]["xor_id"] == stress.complete.iloc[0]["xor_id"]
    assert (
        forward.complete.iloc[0]["xor_sha256"]
        != stress.complete.iloc[0]["xor_sha256"]
    )


def test_signed_xor_orientation_flip_splits_directional_runs_and_reconciles() -> None:
    dates = pd.bdate_range("2005-01-03", periods=5).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 120.0, 80.0, 90.0, 95.0]
    primary = _run(dates, opens, [0, 1, 1, 1, 1], policy="primary")
    comparator = _run(dates, opens, [1, 0, 1, 1, 1], policy="comparator")
    primary_start = ledger.AccountState.initial(
        policy_name="primary", cost_bps=5.0
    )
    comparator_start = ledger.AccountState.initial(
        policy_name="comparator", cost_bps=5.0
    )
    extraction = ledger.extract_signed_xor_episodes(
        primary.ledger,
        comparator.ledger,
        primary_start_state=primary_start,
        comparator_start_state=comparator_start,
    )
    proof = ledger.reconcile_signed_xor(
        primary.ledger,
        comparator.ledger,
        extraction,
        primary_start_state=primary_start,
        comparator_start_state=comparator_start,
    )

    assert len(extraction.complete) == 2
    assert extraction.complete["orientation"].tolist() == [
        "primary_cash_comparator_long",
        "primary_long_comparator_cash",
    ]
    assert "mixed" not in extraction.complete["orientation"].tolist()
    assert extraction.unresolved.empty
    assert math.fsum(extraction.complete["net_signed_log_edge"]) == pytest.approx(
        math.fsum(extraction.components["net_signed_log_edge"]), abs=1e-15
    )
    assert math.fsum(
        extraction.complete["raw_market_component"]
    ) == pytest.approx(
        math.fsum(extraction.components["raw_market_component"]), abs=1e-15
    )
    assert math.fsum(
        extraction.complete["transition_cost_component"]
    ) == pytest.approx(
        math.fsum(extraction.components["transition_cost_component"]), abs=1e-15
    )
    assert abs(proof["reconciliation_error"]) <= 1e-10


def test_signed_xor_reconciliation_rejects_tampered_orientation_hash_and_components() -> None:
    dates = pd.bdate_range("2005-01-03", periods=6).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 105.0, 120.0, 90.0, 80.0, 100.0]
    primary = _run(dates, opens, [1, 0, 0, 1, 1, 1], policy="primary")
    comparator = _run(dates, opens, [1] * len(dates), policy="comparator")
    primary_start = ledger.AccountState.initial(
        policy_name="primary", cost_bps=5.0
    )
    comparator_start = ledger.AccountState.initial(
        policy_name="comparator", cost_bps=5.0
    )
    extracted = ledger.extract_signed_xor_episodes(
        primary.ledger,
        comparator.ledger,
        primary_start_state=primary_start,
        comparator_start_state=comparator_start,
    )
    bad_complete = extracted.complete.copy()
    bad_complete.at[0, "orientation"] = "mixed"
    bad_complete.at[0, "xor_sha256"] = "sha256:" + "0" * 64
    tampered = ledger.XorExtraction(
        components=extracted.components.copy(),
        complete=bad_complete,
        unresolved=extracted.unresolved.copy(),
    )
    with pytest.raises(Error, match="deterministic regeneration"):
        ledger.reconcile_signed_xor(
            primary.ledger,
            comparator.ledger,
            tampered,
            primary_start_state=primary_start,
            comparator_start_state=comparator_start,
        )
    bad_components = extracted.components.copy()
    bad_components.at[1, "net_signed_log_edge"] += 0.01
    tampered_components = ledger.XorExtraction(
        components=bad_components,
        complete=extracted.complete.copy(),
        unresolved=extracted.unresolved.copy(),
    )
    with pytest.raises(Error, match="deterministic regeneration"):
        ledger.reconcile_signed_xor(
            primary.ledger,
            comparator.ledger,
            tampered_components,
            primary_start_state=primary_start,
            comparator_start_state=comparator_start,
        )


def test_unresolved_xor_tail_is_explicit_and_reconciliation_rejects() -> None:
    dates = pd.bdate_range("2005-01-03", periods=4).strftime("%Y-%m-%d").tolist()
    opens = [100.0, 110.0, 95.0, 90.0]
    primary = _run(dates, opens, [0, 0, 0, 0], policy="primary")
    comparator = _run(dates, opens, [1, 1, 1, 1], policy="comparator")
    primary_start = ledger.AccountState.initial(
        policy_name="primary", cost_bps=5.0
    )
    comparator_start = ledger.AccountState.initial(
        policy_name="comparator", cost_bps=5.0
    )
    extraction = ledger.extract_signed_xor_episodes(
        primary.ledger,
        comparator.ledger,
        primary_start_state=primary_start,
        comparator_start_state=comparator_start,
    )

    assert extraction.complete.empty
    assert extraction.unresolved.iloc[0]["status"] == "right_boundary_partial"
    with pytest.raises(Error, match="XOR boundary"):
        ledger.reconcile_signed_xor(
            primary.ledger,
            comparator.ledger,
            extraction,
            primary_start_state=primary_start,
            comparator_start_state=comparator_start,
        )
