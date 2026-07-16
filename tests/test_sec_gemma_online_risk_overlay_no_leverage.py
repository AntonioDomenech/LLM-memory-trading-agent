from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    canonical_sha256,
)
from agent_benchmark import sec_gemma_online_risk_overlay_ledger as ledger_module
from agent_benchmark.sec_gemma_online_risk_overlay_ledger import (
    run_binary_ledger,
)
from agent_benchmark.sec_gemma_online_risk_overlay_no_leverage import (
    SecGemmaOnlineRiskOverlayNoLeverageError,
    validate_sec_gemma_online_risk_overlay_no_leverage_proof,
)


_LEDGER_SCHEMA_VERSION = "aapl-sec-gemma-online-risk-overlay-v2-1-ledger-v1"
_GENESIS = canonical_sha256(
    {
        "schema_version": _LEDGER_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "kind": "ledger_genesis",
    }
)


def _case(
    *,
    cost_bps: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sessions = [
        "2025-01-02",
        "2025-01-03",
        "2025-01-06",
        "2025-01-07",
        "2025-01-08",
        "2025-01-09",
    ]
    opens = [100.0, 102.0, 101.0, 99.0, 100.0, 103.0]
    closes = [101.0, 101.5, 100.0, 100.5, 102.0, 104.0]
    market = [
        {
            "session": session,
            "adjusted_open_hex": adjusted_open.hex(),
            "adjusted_close_hex": adjusted_close.hex(),
        }
        for session, adjusted_open, adjusted_close in zip(
            sessions,
            opens,
            closes,
            strict=True,
        )
    ]
    targets: list[dict[str, Any]] = []
    for session, exposure in zip(
        sessions,
        [1, 1, 0, 0, 1, 1],
        strict=True,
    ):
        body = {
            "session": session,
            "target_exposure": exposure,
        }
        targets.append(
            {**body, "target_row_sha256": canonical_sha256(body)}
        )
    ledger = run_binary_ledger(
        market_rows=market,
        target_rows=targets,
        policy_id="semantic",
        cost_bps=cost_bps,
    )
    return market, targets, ledger


def _rehash_outer(ledger: dict[str, Any]) -> dict[str, Any]:
    ledger["ledger_rows_sha256"] = canonical_sha256(
        ledger["ledger_rows"]
    )
    ledger["terminal_state_sha256"] = canonical_sha256(
        ledger["terminal_state"]
    )
    body = {
        key: ledger[key] for key in ledger if key != "ledger_sha256"
    }
    ledger["ledger_sha256"] = canonical_sha256(body)
    return ledger


def _rehash_all_rows(ledger: dict[str, Any]) -> dict[str, Any]:
    parent = _GENESIS
    for row in ledger["ledger_rows"]:
        row["previous_row_sha256"] = parent
        body = {
            key: row[key] for key in row if key != "ledger_row_sha256"
        }
        row["ledger_row_sha256"] = canonical_sha256(body)
        parent = row["ledger_row_sha256"]
    ledger["terminal_state"]["ledger_tip_sha256"] = parent
    return _rehash_outer(ledger)


def _validate(
    market: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    ledger: dict[str, Any],
    *,
    cost_bps: int | None = None,
) -> dict[str, Any]:
    return validate_sec_gemma_online_risk_overlay_no_leverage_proof(
        ledger,
        expected_ledger_sha256=ledger["ledger_sha256"],
        market_rows=market,
        target_rows=targets,
        policy_id="semantic",
        cost_bps=ledger["cost_bps"] if cost_bps is None else cost_bps,
    )


@pytest.mark.parametrize("cost_bps", [5, 10])
def test_independent_proof_replays_valid_ledger_without_runner(
    cost_bps: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market, targets, ledger = _case(cost_bps=cost_bps)

    def forbidden_runner(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the independent verifier called the runner")

    monkeypatch.setattr(
        ledger_module,
        "run_binary_ledger",
        forbidden_runner,
    )
    proof = _validate(market, targets, ledger)
    proof_body = {
        key: proof[key] for key in proof if key != "proof_sha256"
    }

    assert proof["proof_sha256"] == canonical_sha256(proof_body)
    assert proof["observed_realized_exposures"] == [0, 1]
    assert proof["maximum_realized_exposure"] == 1
    assert proof["minimum_cash_hex"] == 0.0.hex()
    assert proof["minimum_shares_hex"] == 0.0.hex()
    assert proof["open_to_open_return_count"] == len(market) - 1
    assert proof["changing_leg_fill_count"] == 3
    assert proof["same_initial_purchase_semantics"] is True
    assert proof["terminal_close_is_valuation_only"] is True
    assert proof["terminal_close_fill_count"] == 0
    assert proof["shorting"] is False
    assert proof["borrowing"] is False
    assert proof["margin"] is False
    assert proof["hidden_interest"] is False


def test_external_ledger_pin_is_mandatory() -> None:
    market, targets, ledger = _case()

    with pytest.raises(
        SecGemmaOnlineRiskOverlayNoLeverageError,
        match="externally pinned",
    ):
        validate_sec_gemma_online_risk_overlay_no_leverage_proof(
            ledger,
            expected_ledger_sha256="f" * 64,
            market_rows=market,
            target_rows=targets,
            policy_id="semantic",
            cost_bps=5,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "fractional_exposure",
        "negative_cash",
        "leveraged_shares",
        "price_mismatch",
        "target_mismatch",
        "period_factor",
        "drawdown",
        "terminal_open",
        "hidden_interest",
        "margin_field",
    ],
)
def test_fully_rehashed_execution_tampering_is_rejected(
    mutation: str,
) -> None:
    market, targets, original = _case()
    forged = copy.deepcopy(original)
    if mutation == "fractional_exposure":
        forged["ledger_rows"][0]["target_exposure"] = 0.5
    elif mutation == "negative_cash":
        forged["ledger_rows"][2]["cash_hex"] = (-1.0).hex()
    elif mutation == "leveraged_shares":
        shares = float.fromhex(forged["ledger_rows"][0]["shares_hex"])
        forged["ledger_rows"][0]["shares_hex"] = (shares * 2.0).hex()
    elif mutation == "price_mismatch":
        forged["ledger_rows"][1]["adjusted_open_hex"] = (102.5).hex()
    elif mutation == "target_mismatch":
        forged["ledger_rows"][1]["target_exposure"] = 0
    elif mutation == "period_factor":
        factor = float.fromhex(
            forged["ledger_rows"][1]["period_factor_hex"]
        )
        forged["ledger_rows"][1]["period_factor_hex"] = math.nextafter(
            factor,
            math.inf,
        ).hex()
    elif mutation == "drawdown":
        forged["ledger_rows"][1]["drawdown_hex"] = (-0.25).hex()
    elif mutation == "terminal_open":
        forged["terminal_state"]["terminal_open_equity_hex"] = (
            2_000.0
        ).hex()
    elif mutation == "hidden_interest":
        cash = float.fromhex(forged["ledger_rows"][3]["cash_hex"])
        forged["ledger_rows"][3]["cash_hex"] = (cash + 1.0).hex()
    else:
        forged["ledger_rows"][0]["margin_debt_hex"] = 0.0.hex()
    forged = _rehash_all_rows(forged)

    with pytest.raises(SecGemmaOnlineRiskOverlayNoLeverageError):
        _validate(market, targets, forged)


def test_wrong_rehashed_cost_cannot_relabel_five_bps_fills_as_ten() -> None:
    market, targets, original = _case(cost_bps=5)
    forged = copy.deepcopy(original)
    forged["cost_bps"] = 10
    for row in forged["ledger_rows"]:
        row["cost_bps"] = 10
    forged = _rehash_all_rows(forged)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayNoLeverageError,
        match="independent replay",
    ):
        _validate(market, targets, forged, cost_bps=10)


def test_rehashed_terminal_close_liquidation_fiction_is_rejected() -> None:
    market, targets, original = _case(cost_bps=10)
    forged = copy.deepcopy(original)
    shares = float.fromhex(forged["terminal_state"]["shares_hex"])
    terminal_close = float.fromhex(market[-1]["adjusted_close_hex"])
    fictional_sale_proceeds = shares * terminal_close * (1.0 - 0.001)
    forged["terminal_state"][
        "terminal_close_equity_hex"
    ] = fictional_sale_proceeds.hex()
    forged = _rehash_outer(forged)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayNoLeverageError,
        match="terminal close equity",
    ):
        _validate(market, targets, forged)


def test_rehashed_broken_parent_chain_is_rejected() -> None:
    market, targets, original = _case()
    forged = copy.deepcopy(original)
    rows = forged["ledger_rows"]
    rows[2]["previous_row_sha256"] = "f" * 64
    for position in range(2, len(rows)):
        if position > 2:
            rows[position]["previous_row_sha256"] = rows[position - 1][
                "ledger_row_sha256"
            ]
        body = {
            key: rows[position][key]
            for key in rows[position]
            if key != "ledger_row_sha256"
        }
        rows[position]["ledger_row_sha256"] = canonical_sha256(body)
    forged["terminal_state"]["ledger_tip_sha256"] = rows[-1][
        "ledger_row_sha256"
    ]
    forged = _rehash_outer(forged)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayNoLeverageError,
        match="parent hash",
    ):
        _validate(market, targets, forged)


def test_rehashed_target_input_mismatch_is_rejected() -> None:
    market, targets, ledger = _case()
    forged_targets = copy.deepcopy(targets)
    body = {
        "session": forged_targets[1]["session"],
        "target_exposure": 0,
    }
    forged_targets[1] = {
        **body,
        "target_row_sha256": canonical_sha256(body),
    }

    with pytest.raises(
        SecGemmaOnlineRiskOverlayNoLeverageError,
        match="target inputs",
    ):
        _validate(market, forged_targets, ledger)


def test_first_target_must_preserve_initial_purchase_semantics() -> None:
    market, targets, ledger = _case()
    forged_targets = copy.deepcopy(targets)
    body = {
        "session": forged_targets[0]["session"],
        "target_exposure": 0,
    }
    forged_targets[0] = {
        **body,
        "target_row_sha256": canonical_sha256(body),
    }

    with pytest.raises(
        SecGemmaOnlineRiskOverlayNoLeverageError,
        match="genesis target",
    ):
        _validate(market, forged_targets, ledger)
