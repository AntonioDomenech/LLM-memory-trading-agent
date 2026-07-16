from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_ledger import (
    SecGemmaOnlineRiskOverlayLedgerError,
    baseline_open_targets,
    build_baseline_signal_row,
    build_combined_target_rows,
    build_mature_counterfactual_lesson,
    compare_ledgers,
    run_binary_ledger,
    validate_binary_ledger,
    validate_mature_counterfactual_lesson,
)
from agent_benchmark.sec_gemma_online_risk_overlay_policy import (
    POLICY_OVERLAY_SCHEDULE_SCHEMA_VERSION,
)


def _market(count: int = 60) -> list[dict[str, Any]]:
    sessions = [
        stamp.strftime("%Y-%m-%d")
        for stamp in __import__("pandas").bdate_range(
            "2000-01-03", periods=count
        )
    ]
    return [
        {
            "session": session,
            "adjusted_open_hex": (
                100.0 * math.exp(0.001 * position)
            ).hex(),
            "adjusted_close_hex": (
                100.5 * math.exp(0.001 * position)
            ).hex(),
        }
        for position, session in enumerate(sessions)
    ]


def _signals(
    market: list[dict[str, Any]],
    active: set[int] | None = None,
) -> list[dict[str, Any]]:
    active = active or set()
    return [
        build_baseline_signal_row(
            session=row["session"],
            unfiltered_union_signal=position in active,
        )
        for position, row in enumerate(market)
    ]


def _targets(
    market: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    overlays: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return build_combined_target_rows(
        market_rows=market,
        baseline_signals=signals,
        scheduled_overlays=overlays or [],
    )


def _overlay(
    market: list[dict[str, Any]],
    *,
    decision: int,
    accession: str = "a",
    prediction_hash: str = "a" * 64,
) -> dict[str, Any]:
    body = {
        "schema_version": POLICY_OVERLAY_SCHEDULE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "policy_id": "semantic",
        "accession_number": accession,
        "decision_session": market[decision]["session"],
        "prediction_row_sha256": prediction_hash,
        "entry_session_offset": 1,
        "exit_session_offset": LABEL_MATURITY_OFFSET,
        "horizon_sessions": HORIZON_SESSIONS,
    }
    return {**body, "overlay_schedule_sha256": canonical_sha256(body)}


def test_baseline_signal_fills_cash_for_exactly_the_next_open_interval() -> None:
    market = _market(8)
    signals = _signals(market, {2})

    assert baseline_open_targets(market, signals) == [
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
    ]


def test_sec_overlay_is_cash_from_t_plus_1_until_t_plus_21_open() -> None:
    market = _market(40)
    signals = _signals(market)
    decision = 5
    target_stream = _targets(
        market,
        signals,
        [_overlay(market, decision=decision)],
    )
    values = [
        row["target_exposure"] for row in target_stream["target_rows"]
    ]

    assert values[decision] == 1
    assert values[decision + 1 : decision + 21] == [0] * 20
    assert values[decision + 21] == 1
    assert (
        target_stream["overlay_episodes"][0]["realization_status"]
        == "complete"
    )


def test_target_hashes_do_not_change_when_market_prefix_is_extended() -> None:
    market = _market(60)
    short_market = market[:15]
    schedule = _overlay(market, decision=5)
    later_schedule = _overlay(
        market,
        decision=30,
        accession="b",
        prediction_hash="b" * 64,
    )
    short = _targets(
        short_market,
        _signals(short_market),
        [schedule],
    )
    extended = _targets(
        market,
        _signals(market),
        [schedule, later_schedule],
    )

    assert short["target_rows"] == extended["target_rows"][:15]
    assert (
        short["overlay_episodes"][0]["realization_status"]
        == "active_pending_exit"
    )
    assert (
        extended["overlay_episodes"][0]["realization_status"]
        == "complete"
    )
    assert all(
        row["active_overlay_schedule_sha256"]
        == schedule["overlay_schedule_sha256"]
        for row in short["target_rows"][6:]
    )


def test_terminal_state_preserves_baseline_and_sec_pending_entry() -> None:
    market = _market(6)
    signals = _signals(market, {len(market) - 1})
    stream = _targets(
        market,
        signals,
        [_overlay(market, decision=len(market) - 1)],
    )
    pending = stream["terminal_pending_state"]
    baseline = pending["baseline_next_open_action"]
    sec = pending["sec_overlay_pending_boundary"]

    assert baseline["origin_close_session"] == market[-1]["session"]
    assert baseline["next_open_target_exposure"] == 0
    assert baseline["realization_status"] == "pending_next_open"
    assert sec["realization_status"] == "pending_entry"
    assert sec["entry_session"] is None
    assert sec["exit_session"] is None
    assert not any("price" in key for key in sec)


def test_pending_exit_is_resolved_only_when_t_plus_21_open_exists() -> None:
    market = _market(35)
    decision = 5
    schedule = _overlay(market, decision=decision)
    before_exit_market = market[: decision + LABEL_MATURITY_OFFSET]
    through_exit_market = market[: decision + LABEL_MATURITY_OFFSET + 1]
    before = _targets(
        before_exit_market,
        _signals(before_exit_market),
        [schedule],
    )
    through = _targets(
        through_exit_market,
        _signals(through_exit_market),
        [schedule],
    )

    pending = before["terminal_pending_state"][
        "sec_overlay_pending_boundary"
    ]
    assert pending["realization_status"] == "active_pending_exit"
    assert pending["entry_session"] == market[decision + 1]["session"]
    assert pending["exit_session"] is None
    assert through["terminal_pending_state"][
        "sec_overlay_pending_boundary"
    ] is None
    assert (
        through["overlay_episodes"][0]["realization_status"]
        == "complete"
    )
    assert (
        through["target_rows"][decision + LABEL_MATURITY_OFFSET][
            "target_exposure"
        ]
        == 1
    )
    assert before["target_rows"] == through["target_rows"][:-1]


def test_baseline_cash_inside_overlay_does_not_create_an_extra_fill() -> None:
    market = _market(35)
    signals = _signals(market, {8})
    target_stream = _targets(
        market,
        signals,
        [_overlay(market, decision=5)],
    )
    ledger = run_binary_ledger(
        market_rows=market,
        target_rows=target_stream["target_rows"],
        policy_id="semantic",
        cost_bps=10,
    )
    changing = [
        row["ordinal"]
        for row in ledger["ledger_rows"]
        if row["changing_leg"]
    ]

    assert changing == [1, 7, 27]


def test_nonoverlap_rejects_a_second_schedule_before_the_first_exit() -> None:
    market = _market(60)
    signals = _signals(market)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayLedgerError,
        match="overlap",
    ):
        _targets(
            market,
            signals,
            [
                _overlay(market, decision=5),
                _overlay(
                    market,
                    decision=10,
                    accession="b",
                    prediction_hash="b" * 64,
                ),
            ],
        )


@pytest.mark.parametrize("cost_bps", [5, 10])
def test_ledger_is_binary_unleveraged_and_charges_only_changing_legs(
    cost_bps: int,
) -> None:
    market = _market(8)
    signals = _signals(market, {1})
    target_stream = _targets(market, signals)
    ledger = run_binary_ledger(
        market_rows=market,
        target_rows=target_stream["target_rows"],
        policy_id="baseline",
        cost_bps=cost_bps,
    )
    transitions = [row["transition"] for row in ledger["ledger_rows"]]

    assert transitions[:4] == ["BUY", "HOLD_LONG", "SELL", "BUY"]
    assert ledger["terminal_state"]["minimum_cash_hex"] == 0.0.hex()
    assert ledger["terminal_state"]["minimum_shares_hex"] == 0.0.hex()
    assert ledger["terminal_state"]["maximum_exposure"] == 1
    assert all(
        row["target_exposure"] in (0, 1)
        for row in ledger["ledger_rows"]
    )
    assert validate_binary_ledger(
        ledger,
        expected_ledger_sha256=ledger["ledger_sha256"],
        market_rows=market,
        target_rows=target_stream["target_rows"],
        policy_id="baseline",
        cost_bps=cost_bps,
    ) == ledger["ledger_sha256"]


def test_ledger_replay_rejects_target_or_row_tampering() -> None:
    market = _market(8)
    signals = _signals(market)
    target_stream = _targets(market, signals)
    ledger = run_binary_ledger(
        market_rows=market,
        target_rows=target_stream["target_rows"],
        policy_id="semantic",
        cost_bps=5,
    )
    changed = copy.deepcopy(ledger)
    changed["ledger_rows"][2]["cash_hex"] = 1.0.hex()

    with pytest.raises(
        SecGemmaOnlineRiskOverlayLedgerError,
        match="differs from deterministic replay",
    ):
        validate_binary_ledger(
            changed,
            expected_ledger_sha256=ledger["ledger_sha256"],
            market_rows=market,
            target_rows=target_stream["target_rows"],
            policy_id="semantic",
            cost_bps=5,
        )


def test_counterfactual_lesson_is_unavailable_before_t_plus_21() -> None:
    market = _market(25)
    signals = _signals(market)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayLedgerError,
        match="has not matured",
    ):
        build_mature_counterfactual_lesson(
            market_rows=market,
            baseline_signals=signals,
            decision_session=market[5]["session"],
            accession_number="a",
            feature_row_sha256="f" * 64,
            feature_fit_eligible=True,
            as_of_session=market[-1]["session"],
        )


def test_counterfactual_label_uses_baseline_interactions_and_fixed_10bps() -> None:
    market = _market(50)
    signals = _signals(market, {8, 18})
    decision = 5
    maturity = decision + 21
    lesson = build_mature_counterfactual_lesson(
        market_rows=market,
        baseline_signals=signals,
        decision_session=market[decision]["session"],
        accession_number="a",
        feature_row_sha256="f" * 64,
        feature_fit_eligible=True,
        as_of_session=market[maturity]["session"],
    )

    assert lesson["entry_session"] == market[decision + 1]["session"]
    assert lesson["maturity_session"] == market[maturity]["session"]
    assert lesson["cost_bps"] == 10
    assert lesson["train_eligible"] is True
    assert lesson["audit_only"] is False
    assert lesson["binary_overlay_win"] in (0, 1)
    assert math.isfinite(
        float.fromhex(lesson["incremental_log_edge_10bps_hex"])
    )


def test_counterfactual_lesson_validator_rebuilds_and_rejects_tampering() -> None:
    market = _market(35)
    signals = _signals(market, {8})
    decision = 5
    maturity = decision + LABEL_MATURITY_OFFSET
    kwargs = {
        "market_rows": market,
        "baseline_signals": signals,
        "decision_session": market[decision]["session"],
        "accession_number": "a",
        "feature_row_sha256": "f" * 64,
        "feature_fit_eligible": True,
        "as_of_session": market[maturity]["session"],
    }
    lesson = build_mature_counterfactual_lesson(**kwargs)

    assert validate_mature_counterfactual_lesson(
        lesson,
        expected_counterfactual_lesson_sha256=lesson[
            "counterfactual_lesson_sha256"
        ],
        **kwargs,
    ) == lesson["counterfactual_lesson_sha256"]

    changed = copy.deepcopy(lesson)
    changed["binary_overlay_win"] = 1 - changed["binary_overlay_win"]
    with pytest.raises(
        SecGemmaOnlineRiskOverlayLedgerError,
        match="differs from deterministic replay",
    ):
        validate_mature_counterfactual_lesson(
            changed,
            expected_counterfactual_lesson_sha256=lesson[
                "counterfactual_lesson_sha256"
            ],
            **kwargs,
        )


def test_unavailable_feature_lesson_is_audit_only() -> None:
    market = _market(30)
    lesson = build_mature_counterfactual_lesson(
        market_rows=market,
        baseline_signals=_signals(market),
        decision_session=market[2]["session"],
        accession_number="a",
        feature_row_sha256="f" * 64,
        feature_fit_eligible=False,
        as_of_session=market[23]["session"],
    )

    assert lesson["train_eligible"] is False
    assert lesson["audit_only"] is True


def test_strategy_and_buy_hold_comparison_uses_identical_prices() -> None:
    market = _market(35)
    signals = _signals(market)
    strategy_targets = _targets(
        market,
        signals,
        [_overlay(market, decision=5)],
    )
    benchmark_targets = _targets(market, signals)
    benchmark_rows = copy.deepcopy(benchmark_targets["target_rows"])
    for row in benchmark_rows:
        body = {
            **{
                key: value
                for key, value in row.items()
                if key not in {"target_exposure", "target_row_sha256"}
            },
            "target_exposure": 1,
        }
        body["baseline_target_exposure"] = 1
        body["baseline_cash"] = False
        body["sec_overlay_cash"] = False
        body["active_overlay_schedule_sha256"] = None
        row.clear()
        row.update(
            {**body, "target_row_sha256": __import__(
                "agent_benchmark.sec_gemma_online_risk_overlay_contract",
                fromlist=["canonical_sha256"],
            ).canonical_sha256(body)}
        )
    strategy = run_binary_ledger(
        market_rows=market,
        target_rows=strategy_targets["target_rows"],
        policy_id="semantic",
        cost_bps=5,
    )
    benchmark = run_binary_ledger(
        market_rows=market,
        target_rows=benchmark_rows,
        policy_id="aapl_buy_hold",
        cost_bps=5,
    )
    comparison = compare_ledgers(strategy, benchmark)

    assert math.isfinite(
        float.fromhex(comparison["adjusted_open_active_log_edge_hex"])
    )
    assert math.isfinite(
        float.fromhex(
            comparison["terminal_adjusted_close_active_log_edge_hex"]
        )
    )
