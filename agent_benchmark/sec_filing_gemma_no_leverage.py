"""Independent zero-tolerance proof for SEC/Gemma LONG/CASH score ledgers.

The deterministic scorer already emits execution invariants, but those claims
cannot prove themselves.  This module detaches a complete score receipt,
replays every adjusted-open return and position-changing cost independently,
adapts the rows to the repository's existing unleveraged-ledger proof, and
adds the stricter experiment requirements: exposures are the exact integers
``0`` or ``1`` and cash/share/debt identities have zero tolerance.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hmac
import json
import math
import re
from typing import Any, Final

import pandas as pd

from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_scoring import decode_score_float_hex
from agent_benchmark.unleveraged_aapl import assert_unleveraged_ledger


PROOF_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-no-leverage-proof-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SCORE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_sha256",
        "inputs",
        "configuration",
        "ledger_genesis_sha256",
        "ledger_tip_sha256",
        "ledger_row_count",
        "ledger_rows_sha256",
        "ledger",
        "terminal",
        "metrics",
        "score_receipt_sha256",
    }
)
_CONFIGURATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "stage",
        "score_start_session",
        "score_cutoff_session",
        "terminal_convention",
        "selected_candidate_id",
        "selected_variant",
        "cost_bps",
        "initial_capital_hex",
        "initial_strategy_exposure",
        "initial_benchmark_exposure",
        "prior_adjusted_open_hex",
        "stage_boundary_position_semantics",
        "fill_timing",
        "cash_interval",
        "cost_application",
    }
)
_LEDGER_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "row_index",
        "session",
        "period",
        "adjusted_open_hex",
        "holding_exposure_for_return",
        "target_exposure",
        "strategy_position_changed",
        "strategy_wealth_hex",
        "strategy_cash_hex",
        "strategy_shares_hex",
        "strategy_log_increment_hex",
        "benchmark_target_exposure",
        "benchmark_position_changed",
        "benchmark_wealth_hex",
        "benchmark_cash_hex",
        "benchmark_shares_hex",
        "benchmark_log_increment_hex",
        "active_log_increment_hex",
        "margin_debt_hex",
        "previous_ledger_row_sha256",
        "ledger_row_sha256",
    }
)
_TERMINAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "session",
        "convention",
        "valuation_price_hex",
        "strategy_wealth_hex",
        "benchmark_wealth_hex",
        "strategy_log_increment_hex",
        "benchmark_log_increment_hex",
        "active_log_increment_hex",
        "strategy_max_drawdown_hex",
        "benchmark_max_drawdown_hex",
        "terminal_mark_sha256",
    }
)


class SecFilingGemmaNoLeverageError(ValueError):
    """A score receipt cannot prove exact unleveraged LONG/CASH execution."""


def _require_plain_json(value: Any, location: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise SecFilingGemmaNoLeverageError(f"{location} is non-finite")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _require_plain_json(child, f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise SecFilingGemmaNoLeverageError(
                    f"{location} contains a non-string key"
                )
            _require_plain_json(child, f"{location}.{key}")
        return
    raise SecFilingGemmaNoLeverageError(
        f"{location} must contain detached plain JSON values"
    )


def _snapshot(value: Mapping[str, Any], location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecFilingGemmaNoLeverageError(f"{location} must be a plain dict")
    _require_plain_json(value, location)
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        parsed = json.loads(payload.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise SecFilingGemmaNoLeverageError(
            f"{location} is not canonical JSON"
        ) from exc
    if type(parsed) is not dict:
        raise SecFilingGemmaNoLeverageError(f"{location} must be an object")
    return parsed


def _keys(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    if set(value) != set(expected):
        raise SecFilingGemmaNoLeverageError(f"{location} keys changed")


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaNoLeverageError(f"{location} must be lowercase SHA-256")
    return value


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecFilingGemmaNoLeverageError(
            f"{location} must be an integer at least {minimum}"
        )
    return value


def _exposure(value: Any, location: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise SecFilingGemmaNoLeverageError(
            f"{location} must be the exact integer 0 or 1"
        )
    return value


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise SecFilingGemmaNoLeverageError(f"{location} must be a Boolean")
    return value


def _number(value: Any, location: str) -> float:
    try:
        return decode_score_float_hex(value, location)
    except Exception as exc:
        raise SecFilingGemmaNoLeverageError(
            f"{location} is not canonical finite float.hex"
        ) from exc


def _hex(value: float) -> str:
    if not math.isfinite(value):
        raise SecFilingGemmaNoLeverageError("Replayed value became non-finite")
    if value == 0.0:
        value = 0.0
    return value.hex()


def _equal_hex(observed: Any, expected: float, location: str) -> float:
    number = _number(observed, location)
    if _hex(number) != _hex(expected):
        raise SecFilingGemmaNoLeverageError(
            f"{location} differs from exact independent replay"
        )
    return number


def validate_sec_gemma_no_leverage_proof(
    score_receipt: Mapping[str, Any],
    *,
    expected_score_receipt_sha256: str,
) -> dict[str, Any]:
    """Independently replay and prove one externally pinned score ledger."""

    receipt = _snapshot(score_receipt, "score receipt")
    _keys(receipt, _SCORE_KEYS, "score receipt")
    observed_score_hash = _sha256(
        receipt["score_receipt_sha256"], "score receipt hash"
    )
    expected_score_hash = _sha256(
        expected_score_receipt_sha256, "expected score receipt hash"
    )
    score_body = {
        key: receipt[key] for key in receipt if key != "score_receipt_sha256"
    }
    if (
        not hmac.compare_digest(observed_score_hash, canonical_sha256(score_body))
        or not hmac.compare_digest(observed_score_hash, expected_score_hash)
    ):
        raise SecFilingGemmaNoLeverageError(
            "Score receipt is noncanonical or not externally pinned"
        )

    configuration = receipt["configuration"]
    if type(configuration) is not dict:
        raise SecFilingGemmaNoLeverageError("Score configuration must be a mapping")
    _keys(configuration, _CONFIGURATION_KEYS, "score configuration")
    stage = configuration["stage"]
    if stage not in {"development", "intermediate", "final"}:
        raise SecFilingGemmaNoLeverageError("Score stage is invalid")
    cost_bps = _strict_int(configuration["cost_bps"], "cost_bps")
    if cost_bps not in {5, 10}:
        raise SecFilingGemmaNoLeverageError("Cost must be exactly 5 or 10 bps")
    rate = cost_bps / 10_000.0
    initial_capital = _number(
        configuration["initial_capital_hex"], "initial capital"
    )
    if initial_capital <= 0.0:
        raise SecFilingGemmaNoLeverageError("Initial capital must be positive")
    strategy_exposure = _exposure(
        configuration["initial_strategy_exposure"], "initial strategy exposure"
    )
    benchmark_exposure = _exposure(
        configuration["initial_benchmark_exposure"], "initial benchmark exposure"
    )
    prior_open_value = configuration["prior_adjusted_open_hex"]
    prior_open = (
        None
        if prior_open_value is None
        else _number(prior_open_value, "prior adjusted open")
    )
    if stage == "development":
        if prior_open is not None or strategy_exposure != 0 or benchmark_exposure != 0:
            raise SecFilingGemmaNoLeverageError(
                "Development must be the one cash genesis without a prior open"
            )
    elif prior_open is None or prior_open <= 0.0 or benchmark_exposure != 1:
        raise SecFilingGemmaNoLeverageError(
            "Later stages must inherit a positive prior open and LONG benchmark"
        )

    expected_genesis = canonical_sha256(
        {
            "domain": "aapl-sec-gemma-continuous-ledger-genesis-v1",
            "initial_capital_hex": configuration["initial_capital_hex"],
            "cost_bps": cost_bps,
            "terminal_convention": configuration["terminal_convention"],
            "initial_strategy_exposure": strategy_exposure,
            "initial_benchmark_exposure": benchmark_exposure,
            "prior_adjusted_open_hex": configuration["prior_adjusted_open_hex"],
        }
    )
    observed_genesis = _sha256(receipt["ledger_genesis_sha256"], "ledger genesis")
    if not hmac.compare_digest(observed_genesis, expected_genesis):
        raise SecFilingGemmaNoLeverageError(
            "Ledger genesis differs from the exact execution configuration"
        )

    rows = receipt["ledger"]
    if type(rows) is not list or not rows:
        raise SecFilingGemmaNoLeverageError("Score ledger must be a nonempty list")
    row_count = _strict_int(receipt["ledger_row_count"], "ledger row count", minimum=1)
    if row_count != len(rows) or receipt["ledger_rows_sha256"] != canonical_sha256(rows):
        raise SecFilingGemmaNoLeverageError("Score ledger count or rows hash changed")
    parent = expected_genesis
    strategy_wealth = initial_capital
    benchmark_wealth = initial_capital
    previous_strategy_wealth = initial_capital
    previous_benchmark_wealth = initial_capital
    previous_open = prior_open
    prior_session: str | None = None
    adapter_rows: list[dict[str, float | int]] = []

    for index, raw_row in enumerate(rows):
        if type(raw_row) is not dict:
            raise SecFilingGemmaNoLeverageError("Ledger row must be a plain mapping")
        row = raw_row
        _keys(row, _LEDGER_ROW_KEYS, f"ledger row {index}")
        if row["schema_version"] != "aapl-sec-gemma-ledger-row-v1":
            raise SecFilingGemmaNoLeverageError("Ledger row schema changed")
        if _strict_int(row["row_index"], "ledger row index") != index:
            raise SecFilingGemmaNoLeverageError("Ledger rows are reordered or missing")
        session = row["session"]
        if not isinstance(session, str) or (prior_session is not None and session <= prior_session):
            raise SecFilingGemmaNoLeverageError(
                "Ledger sessions must be strictly chronological"
            )
        if row["period"] != session[:4]:
            raise SecFilingGemmaNoLeverageError("Ledger period changed")
        if row["previous_ledger_row_sha256"] != parent:
            raise SecFilingGemmaNoLeverageError("Ledger parent chain changed")
        row_hash = _sha256(row["ledger_row_sha256"], "ledger row hash")
        row_body = {key: row[key] for key in row if key != "ledger_row_sha256"}
        if not hmac.compare_digest(row_hash, canonical_sha256(row_body)):
            raise SecFilingGemmaNoLeverageError("Ledger row hash changed")

        adjusted_open = _number(row["adjusted_open_hex"], "adjusted open")
        if adjusted_open <= 0.0:
            raise SecFilingGemmaNoLeverageError("Adjusted open must be positive")
        holding = _exposure(
            row["holding_exposure_for_return"], "holding exposure"
        )
        if holding != strategy_exposure:
            raise SecFilingGemmaNoLeverageError(
                "Holding exposure differs from the prior exact position"
            )
        if previous_open is not None:
            if strategy_exposure == 1:
                strategy_wealth *= adjusted_open / previous_open
            if benchmark_exposure == 1:
                benchmark_wealth *= adjusted_open / previous_open

        target = _exposure(row["target_exposure"], "target exposure")
        strategy_changed = _boolean(
            row["strategy_position_changed"], "strategy position changed"
        )
        if strategy_changed is not (target != strategy_exposure):
            raise SecFilingGemmaNoLeverageError(
                "Strategy trade flag differs from the exposure transition"
            )
        if strategy_changed:
            strategy_wealth *= (
                1.0 / (1.0 + rate) if target == 1 else 1.0 - rate
            )
            strategy_exposure = target

        benchmark_target = _exposure(
            row["benchmark_target_exposure"], "benchmark target exposure"
        )
        if benchmark_target != 1:
            raise SecFilingGemmaNoLeverageError("Benchmark target must remain LONG")
        benchmark_changed = _boolean(
            row["benchmark_position_changed"], "benchmark position changed"
        )
        if benchmark_changed is not (benchmark_exposure != 1):
            raise SecFilingGemmaNoLeverageError(
                "Benchmark trade flag differs from the exposure transition"
            )
        if benchmark_changed:
            benchmark_wealth *= 1.0 / (1.0 + rate)
            benchmark_exposure = 1

        observed_strategy_wealth = _equal_hex(
            row["strategy_wealth_hex"], strategy_wealth, "strategy wealth"
        )
        observed_benchmark_wealth = _equal_hex(
            row["benchmark_wealth_hex"], benchmark_wealth, "benchmark wealth"
        )
        if observed_strategy_wealth <= 0.0 or observed_benchmark_wealth <= 0.0:
            raise SecFilingGemmaNoLeverageError("Wealth must remain positive")
        expected_strategy_cash = strategy_wealth if strategy_exposure == 0 else 0.0
        expected_strategy_shares = (
            0.0 if strategy_exposure == 0 else strategy_wealth / adjusted_open
        )
        strategy_cash = _equal_hex(
            row["strategy_cash_hex"], expected_strategy_cash, "strategy cash"
        )
        strategy_shares = _equal_hex(
            row["strategy_shares_hex"], expected_strategy_shares, "strategy shares"
        )
        benchmark_cash = _equal_hex(
            row["benchmark_cash_hex"], 0.0, "benchmark cash"
        )
        benchmark_shares = _equal_hex(
            row["benchmark_shares_hex"],
            benchmark_wealth / adjusted_open,
            "benchmark shares",
        )
        margin_debt = _equal_hex(row["margin_debt_hex"], 0.0, "margin debt")
        strategy_log = math.log(strategy_wealth / previous_strategy_wealth)
        benchmark_log = math.log(benchmark_wealth / previous_benchmark_wealth)
        _equal_hex(
            row["strategy_log_increment_hex"],
            strategy_log,
            "strategy log increment",
        )
        _equal_hex(
            row["benchmark_log_increment_hex"],
            benchmark_log,
            "benchmark log increment",
        )
        _equal_hex(
            row["active_log_increment_hex"],
            strategy_log - benchmark_log,
            "active log increment",
        )
        adapter_rows.append(
            {
                "target_exposure": target,
                "new_exposure_after_fill": strategy_exposure,
                "holding_exposure_for_return": holding,
                "cash": strategy_cash,
                "shares": strategy_shares,
                "margin_interest": margin_debt,
            }
        )
        if benchmark_cash != 0.0 or benchmark_shares < 0.0:
            raise SecFilingGemmaNoLeverageError("Benchmark cash/share identity failed")
        parent = row_hash
        prior_session = session
        previous_open = adjusted_open
        previous_strategy_wealth = strategy_wealth
        previous_benchmark_wealth = benchmark_wealth

    if receipt["ledger_tip_sha256"] != parent:
        raise SecFilingGemmaNoLeverageError("Ledger tip changed")
    if configuration["score_start_session"] != rows[0]["session"]:
        raise SecFilingGemmaNoLeverageError("Score start differs from the ledger")
    if configuration["score_cutoff_session"] != rows[-1]["session"]:
        raise SecFilingGemmaNoLeverageError("Score cutoff differs from the ledger")

    terminal = receipt["terminal"]
    if type(terminal) is not dict:
        raise SecFilingGemmaNoLeverageError("Terminal mark must be a mapping")
    _keys(terminal, _TERMINAL_KEYS, "terminal mark")
    terminal_body = {
        key: terminal[key] for key in terminal if key != "terminal_mark_sha256"
    }
    if terminal["terminal_mark_sha256"] != canonical_sha256(terminal_body):
        raise SecFilingGemmaNoLeverageError("Terminal mark hash changed")
    if terminal["session"] != rows[-1]["session"]:
        raise SecFilingGemmaNoLeverageError("Terminal session differs from the ledger")
    convention = configuration["terminal_convention"]
    if terminal["convention"] != convention or convention not in {
        "adjusted_open",
        "terminal_adjusted_close",
    }:
        raise SecFilingGemmaNoLeverageError("Terminal convention changed")
    terminal_price = _number(terminal["valuation_price_hex"], "terminal price")
    last_open = _number(rows[-1]["adjusted_open_hex"], "last adjusted open")
    terminal_strategy_wealth = strategy_wealth
    terminal_benchmark_wealth = benchmark_wealth
    if convention == "adjusted_open":
        if _hex(terminal_price) != _hex(last_open):
            raise SecFilingGemmaNoLeverageError("Open terminal price changed")
    else:
        if terminal_price <= 0.0:
            raise SecFilingGemmaNoLeverageError("Terminal close must be positive")
        if strategy_exposure == 1:
            terminal_strategy_wealth *= terminal_price / last_open
        terminal_benchmark_wealth *= terminal_price / last_open
    _equal_hex(
        terminal["strategy_wealth_hex"],
        terminal_strategy_wealth,
        "terminal strategy wealth",
    )
    _equal_hex(
        terminal["benchmark_wealth_hex"],
        terminal_benchmark_wealth,
        "terminal benchmark wealth",
    )
    terminal_strategy_log = math.log(terminal_strategy_wealth / strategy_wealth)
    terminal_benchmark_log = math.log(terminal_benchmark_wealth / benchmark_wealth)
    _equal_hex(
        terminal["strategy_log_increment_hex"],
        terminal_strategy_log,
        "terminal strategy log increment",
    )
    _equal_hex(
        terminal["benchmark_log_increment_hex"],
        terminal_benchmark_log,
        "terminal benchmark log increment",
    )
    _equal_hex(
        terminal["active_log_increment_hex"],
        terminal_strategy_log - terminal_benchmark_log,
        "terminal active log increment",
    )

    try:
        legacy_proof = assert_unleveraged_ledger(
            pd.DataFrame(adapter_rows), tolerance=0.0
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SecFilingGemmaNoLeverageError(
            "Independent repository no-leverage proof failed"
        ) from exc
    if legacy_proof.get("passed") is not True:
        raise SecFilingGemmaNoLeverageError(
            "Independent repository no-leverage proof did not pass"
        )

    strict_invariants = receipt["metrics"].get("strict_invariants")
    if type(strict_invariants) is not dict or strict_invariants != {
        "permitted_target_exposures": [0, 1],
        "proof_tolerance_hex": _hex(0.0),
        "fractional_exposure_observed": False,
        "short_exposure_observed": False,
        "leverage_observed": False,
        "negative_cash_observed": False,
        "margin_debt_observed": False,
        "same_market_rows_as_benchmark": True,
        "same_adjusted_open_prices_as_benchmark": True,
    }:
        raise SecFilingGemmaNoLeverageError(
            "Score receipt strict-invariant summary differs from independent proof"
        )

    body = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "score_receipt_sha256": observed_score_hash,
        "ledger_genesis_sha256": expected_genesis,
        "ledger_tip_sha256": parent,
        "ledger_rows_sha256": receipt["ledger_rows_sha256"],
        "ledger_row_count": row_count,
        "proof_tolerance_hex": _hex(0.0),
        "permitted_target_exposures": [0, 1],
        "maximum_requested_target": max(
            row["target_exposure"] for row in adapter_rows
        ),
        "maximum_post_fill_exposure": max(
            row["new_exposure_after_fill"] for row in adapter_rows
        ),
        "maximum_holding_exposure": max(
            row["holding_exposure_for_return"] for row in adapter_rows
        ),
        "minimum_cash_hex": _hex(min(row["cash"] for row in adapter_rows)),
        "minimum_shares_hex": _hex(min(row["shares"] for row in adapter_rows)),
        "total_margin_debt_hex": _hex(
            math.fsum(row["margin_interest"] for row in adapter_rows)
        ),
        "margin_interest_applicable": False,
        "exact_binary_exposure": True,
        "exact_cash_share_identity": True,
        "exact_transaction_cost_replay": True,
        "existing_repository_proof_zero_tolerance": True,
        "shorting": False,
        "borrowing": False,
        "authorizes_outcome_access": False,
    }
    return {**copy.deepcopy(body), "proof_sha256": canonical_sha256(body)}


__all__ = [
    "PROOF_SCHEMA_VERSION",
    "SecFilingGemmaNoLeverageError",
    "validate_sec_gemma_no_leverage_proof",
]
