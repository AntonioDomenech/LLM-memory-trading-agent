"""Independent exact no-leverage proof for the v2 binary ledger.

This module intentionally does not import or call the ledger runner.  It
accepts a complete externally pinned ledger plus its exact market and target
inputs, then independently replays every adjusted-open mark and every
position-changing fill.  The proof is strict: no tolerance, fractional
exposure, debt, shorting, mixed cash/share allocation, or terminal-close trade
is permitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import date
import hmac
import json
import math
import re
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    canonical_sha256,
)


PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-no-leverage-proof-v1"
)
_LEDGER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-ledger-v1"
)
_LEDGER_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-ledger-row-v1"
)
_INITIAL_CASH: Final[float] = 1_000.0
_ALLOWED_COST_BPS: Final[frozenset[int]] = frozenset({5, 10})
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_GENESIS_PREVIOUS_ROW_SHA256: Final[str] = canonical_sha256(
    {
        "schema_version": _LEDGER_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "kind": "ledger_genesis",
    }
)
_LEDGER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "policy_id",
        "cost_bps",
        "initial_cash_hex",
        "ledger_rows",
        "ledger_rows_sha256",
        "terminal_state",
        "terminal_state_sha256",
        "ledger_sha256",
    }
)
_LEDGER_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "policy_id",
        "cost_bps",
        "ordinal",
        "session",
        "adjusted_open_hex",
        "adjusted_close_hex",
        "target_row_sha256",
        "prior_exposure",
        "target_exposure",
        "transition",
        "changing_leg",
        "cash_hex",
        "shares_hex",
        "equity_before_fill_hex",
        "equity_open_hex",
        "period_factor_hex",
        "drawdown_hex",
        "previous_row_sha256",
        "ledger_row_sha256",
    }
)
_TERMINAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "last_session",
        "exposure",
        "cash_hex",
        "shares_hex",
        "terminal_open_equity_hex",
        "terminal_close_equity_hex",
        "minimum_cash_hex",
        "minimum_shares_hex",
        "maximum_exposure",
        "ledger_row_count",
        "ledger_tip_sha256",
    }
)


class SecGemmaOnlineRiskOverlayNoLeverageError(ValueError):
    """Raised when the v2 ledger fails exact independent replay."""


def _require_plain_json(value: Any, location: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"{location} is non-finite"
            )
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _require_plain_json(child, f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise SecGemmaOnlineRiskOverlayNoLeverageError(
                    f"{location} contains a non-string key"
                )
            _require_plain_json(child, f"{location}.{key}")
        return
    raise SecGemmaOnlineRiskOverlayNoLeverageError(
        f"{location} must contain detached plain JSON values"
    )


def _snapshot_mapping(
    value: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be a plain dict"
        )
    _require_plain_json(value, location)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        result = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} is not canonical JSON"
        ) from exc
    if type(result) is not dict:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be an object"
        )
    return result


def _snapshot_rows(
    values: Sequence[Mapping[str, Any]],
    location: str,
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be a sequence"
        )
    return [
        _snapshot_mapping(value, f"{location}[{index}]")
        for index, value in enumerate(values)
    ]


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    location: str,
) -> None:
    if set(value) != set(expected):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} keys changed"
        )


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be an ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be an integer at least {minimum}"
        )
    return value


def _exposure(value: Any, location: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be the exact integer 0 or 1"
        )
    return value


def _boolean(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be a Boolean"
        )
    return value


def _float_hex(
    value: Any,
    location: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be canonical float.hex text"
        )
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be canonical float.hex text"
        ) from exc
    if not math.isfinite(result) or result.hex() != value:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be a canonical finite float"
        )
    if positive and result <= 0.0:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be positive"
        )
    if nonnegative and result < 0.0:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} must be nonnegative"
        )
    return result


def _equal_float_hex(value: Any, expected: float, location: str) -> float:
    observed = _float_hex(value, location)
    if not math.isfinite(expected) or observed.hex() != expected.hex():
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            f"{location} differs from exact independent replay"
        )
    return observed


def _validate_market_rows(
    market_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _snapshot_rows(market_rows, "market_rows")
    if not rows:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "market_rows must not be empty"
        )
    previous_session: str | None = None
    for index, row in enumerate(rows):
        if set(row) != {
            "session",
            "adjusted_open_hex",
            "adjusted_close_hex",
        }:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"market_rows[{index}] keys changed"
            )
        session = _iso_date(row["session"], f"market_rows[{index}].session")
        if previous_session is not None and session <= previous_session:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                "market sessions must be strictly increasing"
            )
        _float_hex(
            row["adjusted_open_hex"],
            f"market_rows[{index}].adjusted_open_hex",
            positive=True,
        )
        _float_hex(
            row["adjusted_close_hex"],
            f"market_rows[{index}].adjusted_close_hex",
            positive=True,
        )
        previous_session = session
    return rows


def _validate_target_rows(
    target_rows: Sequence[Mapping[str, Any]],
    market_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _snapshot_rows(target_rows, "target_rows")
    if len(rows) != len(market_rows):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "target row count differs from market row count"
        )
    required = {"session", "target_exposure", "target_row_sha256"}
    for index, (row, market) in enumerate(
        zip(rows, market_rows, strict=True)
    ):
        if not required.issubset(row):
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"target_rows[{index}] is missing a required field"
            )
        if row["session"] != market["session"]:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                "target rows are not aligned to market sessions"
            )
        _exposure(
            row["target_exposure"],
            f"target_rows[{index}].target_exposure",
        )
        observed_hash = _sha256(
            row["target_row_sha256"],
            f"target_rows[{index}].target_row_sha256",
        )
        body = {
            key: row[key]
            for key in row
            if key != "target_row_sha256"
        }
        if not hmac.compare_digest(observed_hash, canonical_sha256(body)):
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"target_rows[{index}] self-hash changed"
            )
    if rows[0]["target_exposure"] != 1:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "genesis target must purchase AAPL at the first adjusted open"
        )
    return rows


def validate_sec_gemma_online_risk_overlay_no_leverage_proof(
    ledger: Mapping[str, Any],
    *,
    expected_ledger_sha256: str,
    market_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    policy_id: str,
    cost_bps: int,
) -> dict[str, Any]:
    """Independently replay one externally pinned binary LONG/CASH ledger."""

    receipt = _snapshot_mapping(ledger, "ledger")
    _exact_keys(receipt, _LEDGER_KEYS, "ledger")
    observed_ledger_hash = _sha256(
        receipt["ledger_sha256"], "ledger.ledger_sha256"
    )
    expected_ledger_hash = _sha256(
        expected_ledger_sha256, "expected_ledger_sha256"
    )
    ledger_body = {
        key: receipt[key] for key in receipt if key != "ledger_sha256"
    }
    if (
        not hmac.compare_digest(
            observed_ledger_hash,
            canonical_sha256(ledger_body),
        )
        or not hmac.compare_digest(
            observed_ledger_hash,
            expected_ledger_hash,
        )
    ):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "ledger is noncanonical or not externally pinned"
        )
    if (
        receipt["schema_version"] != _LEDGER_SCHEMA_VERSION
        or receipt["contract_version"] != CONTRACT_VERSION
        or receipt["contract_sha256"] != CONTRACT_SHA256
    ):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "ledger schema or contract changed"
        )
    if not isinstance(policy_id, str) or not policy_id:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "policy_id is invalid"
        )
    if receipt["policy_id"] != policy_id:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "ledger policy differs from the exact policy input"
        )
    if type(cost_bps) is not int or cost_bps not in _ALLOWED_COST_BPS:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "cost_bps must be the exact integer 5 or 10"
        )
    if receipt["cost_bps"] != cost_bps:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "ledger cost differs from the exact cost input"
        )
    if receipt["initial_cash_hex"] != _INITIAL_CASH.hex():
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "ledger initial cash or purchase semantics changed"
        )

    market = _validate_market_rows(market_rows)
    targets = _validate_target_rows(target_rows, market)
    rows = receipt["ledger_rows"]
    if type(rows) is not list or len(rows) != len(market):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "ledger row count differs from exact inputs"
        )
    if receipt["ledger_rows_sha256"] != canonical_sha256(rows):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "ledger rows hash changed"
        )

    rate = cost_bps / 10_000.0
    cash = _INITIAL_CASH
    shares = 0.0
    exposure = 0
    equity = _INITIAL_CASH
    running_peak = _INITIAL_CASH
    previous_equity = _INITIAL_CASH
    previous_open: float | None = None
    parent_hash = _GENESIS_PREVIOUS_ROW_SHA256
    minimum_cash = math.inf
    minimum_shares = math.inf
    maximum_exposure = 0
    realized_exposures: set[int] = set()
    changing_leg_count = 0
    buy_count = 0
    sell_count = 0
    hold_cash_count = 0
    hold_long_count = 0
    open_to_open_return_count = 0

    for index, (row_value, market_row, target_row) in enumerate(
        zip(rows, market, targets, strict=True),
        start=1,
    ):
        if type(row_value) is not dict:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} must be a plain dict"
            )
        row = row_value
        _exact_keys(row, _LEDGER_ROW_KEYS, f"ledger row {index}")
        if (
            row["schema_version"] != _LEDGER_ROW_SCHEMA_VERSION
            or row["contract_version"] != CONTRACT_VERSION
            or row["contract_sha256"] != CONTRACT_SHA256
        ):
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} schema or contract changed"
            )
        if row["policy_id"] != policy_id or row["cost_bps"] != cost_bps:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} policy or cost changed"
            )
        if _strict_int(
            row["ordinal"],
            f"ledger row {index}.ordinal",
            minimum=1,
        ) != index:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                "ledger rows are reordered or missing"
            )
        if (
            row["session"] != market_row["session"]
            or row["adjusted_open_hex"] != market_row["adjusted_open_hex"]
            or row["adjusted_close_hex"] != market_row["adjusted_close_hex"]
        ):
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} differs from exact market inputs"
            )
        if row["target_row_sha256"] != target_row["target_row_sha256"]:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} differs from exact target inputs"
            )
        prior_exposure = _exposure(
            row["prior_exposure"],
            f"ledger row {index}.prior_exposure",
        )
        target_exposure = _exposure(
            row["target_exposure"],
            f"ledger row {index}.target_exposure",
        )
        if (
            prior_exposure != exposure
            or target_exposure != target_row["target_exposure"]
        ):
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} exposure differs from exact replay"
            )
        changed = exposure != target_exposure
        if _boolean(
            row["changing_leg"],
            f"ledger row {index}.changing_leg",
        ) is not changed:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} changing-leg flag changed"
            )

        adjusted_open = _float_hex(
            market_row["adjusted_open_hex"],
            f"market row {index}.adjusted_open_hex",
            positive=True,
        )
        if previous_open is not None:
            open_return_factor = adjusted_open / previous_open
            if (
                not math.isfinite(open_return_factor)
                or open_return_factor <= 0.0
            ):
                raise SecGemmaOnlineRiskOverlayNoLeverageError(
                    f"ledger row {index} open-to-open return is invalid"
                )
            open_to_open_return_count += 1
        equity_before_fill = cash + shares * adjusted_open
        if not math.isfinite(equity_before_fill) or equity_before_fill <= 0.0:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} pre-fill equity is invalid"
            )
        _equal_float_hex(
            row["equity_before_fill_hex"],
            equity_before_fill,
            f"ledger row {index}.equity_before_fill_hex",
        )

        expected_transition = (
            "HOLD_LONG" if exposure == 1 else "HOLD_CASH"
        )
        if changed:
            changing_leg_count += 1
            if target_exposure == 0:
                cash = shares * adjusted_open * (1.0 - rate)
                shares = 0.0
                expected_transition = "SELL"
                sell_count += 1
            else:
                shares = cash / (adjusted_open * (1.0 + rate))
                cash = 0.0
                expected_transition = "BUY"
                buy_count += 1
        elif exposure == 0:
            hold_cash_count += 1
        else:
            hold_long_count += 1
        if row["transition"] != expected_transition:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} transition differs from exact replay"
            )
        exposure = target_exposure
        equity = cash + shares * adjusted_open
        if (
            not math.isfinite(cash)
            or not math.isfinite(shares)
            or not math.isfinite(equity)
            or cash < 0.0
            or shares < 0.0
            or equity <= 0.0
        ):
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} shorts, borrows, or has negative cash"
            )
        if exposure == 0:
            if shares != 0.0 or cash <= 0.0:
                raise SecGemmaOnlineRiskOverlayNoLeverageError(
                    f"ledger row {index} realized CASH exposure is not exact"
                )
            realized_exposure = 0
        else:
            if cash != 0.0 or shares <= 0.0:
                raise SecGemmaOnlineRiskOverlayNoLeverageError(
                    f"ledger row {index} realized LONG exposure is not exact"
                )
            realized_exposure = 1
        realized_exposures.add(realized_exposure)

        _equal_float_hex(
            row["cash_hex"],
            cash,
            f"ledger row {index}.cash_hex",
        )
        _equal_float_hex(
            row["shares_hex"],
            shares,
            f"ledger row {index}.shares_hex",
        )
        _equal_float_hex(
            row["equity_open_hex"],
            equity,
            f"ledger row {index}.equity_open_hex",
        )
        period_factor = equity / previous_equity
        if not math.isfinite(period_factor) or period_factor <= 0.0:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} period factor is invalid"
            )
        _equal_float_hex(
            row["period_factor_hex"],
            period_factor,
            f"ledger row {index}.period_factor_hex",
        )
        running_peak = max(running_peak, equity)
        drawdown = equity / running_peak - 1.0
        _equal_float_hex(
            row["drawdown_hex"],
            drawdown,
            f"ledger row {index}.drawdown_hex",
        )

        if row["previous_row_sha256"] != parent_hash:
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} parent hash changed"
            )
        row_hash = _sha256(
            row["ledger_row_sha256"],
            f"ledger row {index}.ledger_row_sha256",
        )
        row_body = {
            key: row[key] for key in row if key != "ledger_row_sha256"
        }
        if not hmac.compare_digest(row_hash, canonical_sha256(row_body)):
            raise SecGemmaOnlineRiskOverlayNoLeverageError(
                f"ledger row {index} self-hash changed"
            )
        parent_hash = row_hash
        previous_equity = equity
        previous_open = adjusted_open
        minimum_cash = min(minimum_cash, cash)
        minimum_shares = min(minimum_shares, shares)
        maximum_exposure = max(maximum_exposure, exposure)

    first_row = rows[0]
    if not (
        first_row["prior_exposure"] == 0
        and first_row["target_exposure"] == 1
        and first_row["changing_leg"] is True
        and first_row["transition"] == "BUY"
        and buy_count >= 1
    ):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "initial purchase semantics changed"
        )

    terminal = receipt["terminal_state"]
    if type(terminal) is not dict:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "terminal_state must be a plain dict"
        )
    _exact_keys(terminal, _TERMINAL_KEYS, "terminal_state")
    terminal_hash = _sha256(
        receipt["terminal_state_sha256"],
        "terminal_state_sha256",
    )
    if not hmac.compare_digest(terminal_hash, canonical_sha256(terminal)):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "terminal state self-hash changed"
        )
    terminal_close = _float_hex(
        market[-1]["adjusted_close_hex"],
        "terminal adjusted close",
        positive=True,
    )
    terminal_close_equity = cash + shares * terminal_close
    if not math.isfinite(terminal_close_equity) or terminal_close_equity <= 0.0:
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "terminal close equity is invalid"
        )
    if (
        terminal["last_session"] != market[-1]["session"]
        or _exposure(terminal["exposure"], "terminal exposure") != exposure
        or _strict_int(
            terminal["maximum_exposure"],
            "terminal maximum exposure",
        )
        != maximum_exposure
        or _strict_int(
            terminal["ledger_row_count"],
            "terminal ledger row count",
            minimum=1,
        )
        != len(rows)
        or terminal["ledger_tip_sha256"] != parent_hash
    ):
        raise SecGemmaOnlineRiskOverlayNoLeverageError(
            "terminal state identity differs from exact replay"
        )
    _equal_float_hex(terminal["cash_hex"], cash, "terminal cash")
    _equal_float_hex(terminal["shares_hex"], shares, "terminal shares")
    _equal_float_hex(
        terminal["terminal_open_equity_hex"],
        equity,
        "terminal open equity",
    )
    _equal_float_hex(
        terminal["terminal_close_equity_hex"],
        terminal_close_equity,
        "terminal close equity",
    )
    _equal_float_hex(
        terminal["minimum_cash_hex"],
        minimum_cash,
        "terminal minimum cash",
    )
    _equal_float_hex(
        terminal["minimum_shares_hex"],
        minimum_shares,
        "terminal minimum shares",
    )

    proof_body = {
        "schema_version": PROOF_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "ledger_sha256": observed_ledger_hash,
        "market_rows_sha256": canonical_sha256(market),
        "target_rows_sha256": canonical_sha256(targets),
        "policy_id": policy_id,
        "cost_bps": cost_bps,
        "initial_cash_hex": _INITIAL_CASH.hex(),
        "ledger_row_count": len(rows),
        "ledger_tip_sha256": parent_hash,
        "proof_tolerance_hex": 0.0.hex(),
        "permitted_exposures": [0, 1],
        "observed_realized_exposures": sorted(realized_exposures),
        "minimum_realized_exposure": min(realized_exposures),
        "maximum_realized_exposure": max(realized_exposures),
        "minimum_cash_hex": minimum_cash.hex(),
        "minimum_shares_hex": minimum_shares.hex(),
        "maximum_target_exposure": maximum_exposure,
        "open_to_open_return_count": open_to_open_return_count,
        "changing_leg_fill_count": changing_leg_count,
        "buy_fill_count": buy_count,
        "sell_fill_count": sell_count,
        "hold_cash_count": hold_cash_count,
        "hold_long_count": hold_long_count,
        "terminal_open_equity_hex": equity.hex(),
        "terminal_close_equity_hex": terminal_close_equity.hex(),
        "terminal_close_valuation_price_hex": terminal_close.hex(),
        "terminal_close_is_valuation_only": True,
        "terminal_close_fill_count": 0,
        "same_initial_purchase_semantics": True,
        "exact_open_to_open_return_replay": True,
        "exact_transaction_cost_replay": True,
        "exact_cash_share_equity_identity": True,
        "exact_binary_target_and_realized_exposure": True,
        "hash_chain_verified": True,
        "shorting": False,
        "borrowing": False,
        "margin": False,
        "negative_cash": False,
        "hidden_interest": False,
        "passed": True,
        "authorizes_outcome_access": False,
    }
    return {
        **copy.deepcopy(proof_body),
        "proof_sha256": canonical_sha256(proof_body),
    }


__all__ = [
    "PROOF_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayNoLeverageError",
    "validate_sec_gemma_online_risk_overlay_no_leverage_proof",
]
