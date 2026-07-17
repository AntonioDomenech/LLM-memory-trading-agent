"""Pure binary LONG/CASH ledger for the SEC/Gemma online risk overlay.

The module has no filesystem, network, clock, SEC, model, or feature-building
authority.  It consumes already authenticated adjusted-open rows and sealed
close-time actions.  All persisted floating-point values use canonical
``float.hex`` strings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from datetime import date
import hmac
import math
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    POSITIVE_EDGE_TOLERANCE,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_policy import (
    POLICY_OVERLAY_SCHEDULE_SCHEMA_VERSION,
)


LEDGER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-ledger-v1"
)
LEDGER_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-ledger-row-v1"
)
COUNTERFACTUAL_LESSON_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-counterfactual-lesson-v1"
)
OVERLAY_EPISODE_OBSERVATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-episode-observation-v1"
)
TARGET_STREAM_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-target-stream-v2"
)
TERMINAL_PENDING_STATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-terminal-pending-state-v1"
)
INITIAL_CASH: Final[float] = 1_000.0
ALLOWED_COST_BPS: Final[frozenset[int]] = frozenset({5, 10})
GENESIS_PREVIOUS_ROW_SHA256: Final[str] = canonical_sha256(
    {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "kind": "ledger_genesis",
    }
)


class SecGemmaOnlineRiskOverlayLedgerError(ValueError):
    """Raised when an action, price, ledger, or lesson is not exact."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be a mapping"
        )
    return value


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be an ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _float_hex(value: Any, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be a finite number"
        )
    return number.hex()


def _decode_float_hex(
    value: Any,
    location: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be canonical float.hex text"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be canonical float.hex text"
        ) from exc
    if not math.isfinite(number) or number.hex() != value:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be a canonical finite float"
        )
    if positive and number <= 0.0:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be positive"
        )
    if nonnegative and number < 0.0:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be nonnegative"
        )
    return number


def _binary(value: Any, location: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            f"{location} must be canonical integer 0 or 1"
        )
    return value


def _cost_bps(value: Any) -> int:
    if type(value) is not int or value not in ALLOWED_COST_BPS:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "cost_bps must be canonical integer 5 or 10"
        )
    return value


def _canonical_market_rows(
    market_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(market_rows, (str, bytes)) or not isinstance(
        market_rows, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "market rows must be a sequence"
        )
    result: list[dict[str, Any]] = []
    previous: str | None = None
    for ordinal, raw in enumerate(market_rows, start=1):
        row = dict(_mapping(raw, f"market row {ordinal}"))
        if set(row) != {
            "session",
            "adjusted_open_hex",
            "adjusted_close_hex",
        }:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "market row keys changed"
            )
        session = _iso_date(row["session"], f"market row {ordinal}.session")
        if previous is not None and session <= previous:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "market sessions must be strictly increasing"
            )
        _decode_float_hex(
            row["adjusted_open_hex"],
            f"market row {ordinal}.adjusted_open_hex",
            positive=True,
        )
        _decode_float_hex(
            row["adjusted_close_hex"],
            f"market row {ordinal}.adjusted_close_hex",
            positive=True,
        )
        result.append(copy.deepcopy(row))
        previous = session
    if not result:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "market rows must not be empty"
        )
    return result


def _canonical_baseline_signals(
    market_rows: Sequence[Mapping[str, Any]],
    baseline_signals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(baseline_signals, (str, bytes)) or not isinstance(
        baseline_signals, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "baseline signals must be a sequence"
        )
    if len(baseline_signals) != len(market_rows):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "baseline signal count differs from market row count"
        )
    result: list[dict[str, Any]] = []
    for ordinal, (market, raw) in enumerate(
        zip(market_rows, baseline_signals, strict=True),
        start=1,
    ):
        row = dict(_mapping(raw, f"baseline signal {ordinal}"))
        if set(row) != {
            "session",
            "unfiltered_union_signal",
            "baseline_signal_sha256",
        }:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "baseline signal keys changed"
            )
        if row["session"] != market["session"]:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "baseline signals are not aligned to market sessions"
            )
        if type(row["unfiltered_union_signal"]) is not bool:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "baseline signal must be Boolean"
            )
        observed = _sha256(
            row["baseline_signal_sha256"],
            f"baseline signal {ordinal}.baseline_signal_sha256",
        )
        body = {
            "session": row["session"],
            "unfiltered_union_signal": row["unfiltered_union_signal"],
        }
        if not hmac.compare_digest(observed, canonical_sha256(body)):
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "baseline signal self-hash changed"
            )
        result.append(copy.deepcopy(row))
    return result


def build_baseline_signal_row(
    *,
    session: str,
    unfiltered_union_signal: bool,
) -> dict[str, Any]:
    """Build one hash-bound close-time baseline signal."""

    session = _iso_date(session, "baseline session")
    if type(unfiltered_union_signal) is not bool:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "unfiltered_union_signal must be Boolean"
        )
    body = {
        "session": session,
        "unfiltered_union_signal": unfiltered_union_signal,
    }
    return {**body, "baseline_signal_sha256": canonical_sha256(body)}


def baseline_open_targets(
    market_rows: Sequence[Mapping[str, Any]],
    baseline_signals: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Return the baseline exposure at each open.

    A close signal at row ``t`` schedules CASH at open ``t+1``.  The already
    canonical one-session signal stream therefore returns to LONG at ``t+2``.
    """

    market = _canonical_market_rows(market_rows)
    signals = _canonical_baseline_signals(market, baseline_signals)
    targets = [1]
    targets.extend(
        0 if signals[position - 1]["unfiltered_union_signal"] else 1
        for position in range(1, len(signals))
    )
    return targets


def build_combined_target_rows(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    baseline_signals: Sequence[Mapping[str, Any]],
    scheduled_overlays: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine one-session baseline exits with non-overlapping SEC overlays."""

    market = _canonical_market_rows(market_rows)
    signals = _canonical_baseline_signals(market, baseline_signals)
    sessions = [row["session"] for row in market]
    session_to_position = {
        session: position for position, session in enumerate(sessions)
    }
    baseline_targets = baseline_open_targets(market, signals)
    if isinstance(scheduled_overlays, (str, bytes)) or not isinstance(
        scheduled_overlays, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "scheduled overlays must be a sequence"
        )

    overlay_positions: list[tuple[int, int, dict[str, Any]]] = []
    episode_rows: list[dict[str, Any]] = []
    previous_exit = -1
    seen_accessions: set[str] = set()
    seen_schedule_hashes: set[str] = set()
    for ordinal, raw in enumerate(scheduled_overlays, start=1):
        value = dict(_mapping(raw, f"scheduled overlay {ordinal}"))
        if set(value) != {
            "schema_version",
            "contract_version",
            "contract_sha256",
            "policy_id",
            "accession_number",
            "decision_session",
            "prediction_row_sha256",
            "entry_session_offset",
            "exit_session_offset",
            "horizon_sessions",
            "overlay_schedule_sha256",
        }:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled overlay keys changed"
            )
        if (
            value["schema_version"]
            != POLICY_OVERLAY_SCHEDULE_SCHEMA_VERSION
            or value["contract_version"] != CONTRACT_VERSION
            or value["contract_sha256"] != CONTRACT_SHA256
        ):
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled overlay schema or contract changed"
            )
        policy_id = value["policy_id"]
        if not isinstance(policy_id, str) or not policy_id:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled overlay policy_id is invalid"
            )
        accession = value["accession_number"]
        if (
            not isinstance(accession, str)
            or not accession
            or accession in seen_accessions
        ):
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled overlay accession is invalid or duplicated"
            )
        seen_accessions.add(accession)
        decision = _iso_date(
            value["decision_session"],
            f"scheduled overlay {ordinal}.decision_session",
        )
        prediction_hash = _sha256(
            value["prediction_row_sha256"],
            f"scheduled overlay {ordinal}.prediction_row_sha256",
        )
        if (
            type(value["entry_session_offset"]) is not int
            or value["entry_session_offset"] != 1
            or type(value["exit_session_offset"]) is not int
            or value["exit_session_offset"] != LABEL_MATURITY_OFFSET
            or type(value["horizon_sessions"]) is not int
            or value["horizon_sessions"] != HORIZON_SESSIONS
        ):
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled overlay horizon changed"
            )
        observed_schedule_hash = _sha256(
            value["overlay_schedule_sha256"],
            f"scheduled overlay {ordinal}.overlay_schedule_sha256",
        )
        schedule_body = {
            key: value[key]
            for key in value
            if key != "overlay_schedule_sha256"
        }
        if not hmac.compare_digest(
            observed_schedule_hash,
            canonical_sha256(schedule_body),
        ):
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled overlay self-hash changed"
            )
        if observed_schedule_hash in seen_schedule_hashes:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled overlay hash is duplicated"
            )
        seen_schedule_hashes.add(observed_schedule_hash)
        if decision not in session_to_position:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled overlay decision is outside the market prefix"
            )
        decision_position = session_to_position[decision]
        entry_position = decision_position + 1
        exit_position = decision_position + LABEL_MATURITY_OFFSET
        if entry_position <= previous_exit:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "scheduled SEC overlays overlap or extend one another"
            )
        previous_exit = exit_position
        entry_session = (
            sessions[entry_position] if entry_position < len(sessions) else None
        )
        exit_session = (
            sessions[exit_position] if exit_position < len(sessions) else None
        )
        status = (
            "pending_entry"
            if entry_session is None
            else "complete"
            if exit_session is not None
            else "active_pending_exit"
        )
        episode_body = {
            "schema_version": OVERLAY_EPISODE_OBSERVATION_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "overlay_schedule_sha256": observed_schedule_hash,
            "policy_id": policy_id,
            "accession_number": accession,
            "decision_session": decision,
            "entry_session": entry_session,
            "exit_session": exit_session,
            "entry_position": entry_position,
            "exit_position": exit_position,
            "horizon_sessions": HORIZON_SESSIONS,
            "realization_status": status,
            "prediction_row_sha256": prediction_hash,
            "market_prefix_last_session": sessions[-1],
            "market_sessions_sha256": canonical_sha256(sessions),
        }
        episode = {
            **episode_body,
            "overlay_episode_observation_sha256": canonical_sha256(
                episode_body
            ),
        }
        episode_rows.append(episode)
        overlay_positions.append((entry_position, exit_position, value))

    target_rows: list[dict[str, Any]] = []
    active_schedule: dict[str, Any] | None = None
    for position, (market_row, baseline_target) in enumerate(
        zip(market, baseline_targets, strict=True)
    ):
        active_schedule = None
        for entry, exit_, schedule in overlay_positions:
            if entry <= position < exit_:
                active_schedule = schedule
                break
        overlay_cash = active_schedule is not None
        target = 0 if baseline_target == 0 or overlay_cash else 1
        body = {
            "session": market_row["session"],
            "baseline_target_exposure": baseline_target,
            "baseline_cash": baseline_target == 0,
            "sec_overlay_cash": overlay_cash,
            "active_overlay_schedule_sha256": (
                None
                if active_schedule is None
                else active_schedule["overlay_schedule_sha256"]
            ),
            "target_exposure": target,
        }
        target_rows.append(
            {**body, "target_row_sha256": canonical_sha256(body)}
        )
    baseline_pending_body = {
        "origin_close_session": sessions[-1],
        "baseline_signal_sha256": signals[-1]["baseline_signal_sha256"],
        "next_open_target_exposure": (
            0 if signals[-1]["unfiltered_union_signal"] else 1
        ),
        "realization_status": "pending_next_open",
    }
    baseline_pending = {
        **baseline_pending_body,
        "baseline_pending_action_sha256": canonical_sha256(
            baseline_pending_body
        ),
    }
    pending_sec = [
        episode
        for episode in episode_rows
        if episode["realization_status"]
        in {"pending_entry", "active_pending_exit"}
    ]
    if len(pending_sec) > 1:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "more than one SEC overlay boundary is pending"
        )
    terminal_pending_body = {
        "schema_version": TERMINAL_PENDING_STATE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "as_of_session": sessions[-1],
        "baseline_next_open_action": baseline_pending,
        "sec_overlay_pending_boundary": (
            None if not pending_sec else pending_sec[0]
        ),
    }
    terminal_pending_state = {
        **terminal_pending_body,
        "terminal_pending_state_sha256": canonical_sha256(
            terminal_pending_body
        ),
    }
    body = {
        "schema_version": TARGET_STREAM_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "target_rows": target_rows,
        "target_rows_sha256": canonical_sha256(target_rows),
        "overlay_episodes": episode_rows,
        "overlay_episodes_sha256": canonical_sha256(episode_rows),
        "terminal_pending_state": terminal_pending_state,
        "terminal_pending_state_sha256": terminal_pending_state[
            "terminal_pending_state_sha256"
        ],
    }
    return {**body, "target_stream_sha256": canonical_sha256(body)}


@dataclass(frozen=True)
class _Account:
    cash: float
    shares: float
    exposure: int
    equity: float
    running_peak: float


def _advance(
    account: _Account,
    *,
    adjusted_open: float,
    target: int,
    cost_bps: int,
) -> tuple[_Account, dict[str, Any]]:
    target = _binary(target, "target exposure")
    cost = _cost_bps(cost_bps) / 10_000.0
    equity_before = account.cash + account.shares * adjusted_open
    if not math.isfinite(equity_before) or equity_before <= 0.0:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "pre-fill equity is invalid"
        )
    cash = account.cash
    shares = account.shares
    transition = "HOLD_LONG" if account.exposure == 1 else "HOLD_CASH"
    if account.exposure != target:
        if target == 0:
            cash = shares * adjusted_open * (1.0 - cost)
            shares = 0.0
            transition = "SELL"
        else:
            shares = cash / (adjusted_open * (1.0 + cost))
            cash = 0.0
            transition = "BUY"
    equity_after = cash + shares * adjusted_open
    if (
        not math.isfinite(cash)
        or not math.isfinite(shares)
        or not math.isfinite(equity_after)
        or cash < 0.0
        or shares < 0.0
        or equity_after <= 0.0
    ):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "post-fill account is invalid"
        )
    running_peak = max(account.running_peak, equity_after)
    drawdown = equity_after / running_peak - 1.0
    result = _Account(
        cash=cash,
        shares=shares,
        exposure=target,
        equity=equity_after,
        running_peak=running_peak,
    )
    diagnostics = {
        "transition": transition,
        "equity_before_fill": equity_before,
        "equity_after_fill": equity_after,
        "drawdown": drawdown,
        "changing_leg": account.exposure != target,
    }
    return result, diagnostics


def run_binary_ledger(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    policy_id: str,
    cost_bps: int,
) -> dict[str, Any]:
    """Run one exact continuous 0/1 account from the common genesis cash."""

    market = _canonical_market_rows(market_rows)
    if not isinstance(policy_id, str) or not policy_id:
        raise SecGemmaOnlineRiskOverlayLedgerError("policy_id is invalid")
    cost_bps = _cost_bps(cost_bps)
    if isinstance(target_rows, (str, bytes)) or not isinstance(
        target_rows, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "target rows must be a sequence"
        )
    if len(target_rows) != len(market):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "target row count differs from market row count"
        )
    targets: list[dict[str, Any]] = []
    for ordinal, (market_row, raw) in enumerate(
        zip(market, target_rows, strict=True),
        start=1,
    ):
        row = dict(_mapping(raw, f"target row {ordinal}"))
        required = {
            "session",
            "target_exposure",
            "target_row_sha256",
        }
        if not required.issubset(row):
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "target row is missing an exact required field"
            )
        if row["session"] != market_row["session"]:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "target rows are not aligned to market sessions"
            )
        _binary(row["target_exposure"], "target row exposure")
        observed = _sha256(
            row["target_row_sha256"],
            f"target row {ordinal}.target_row_sha256",
        )
        body = {
            key: row[key]
            for key in row
            if key != "target_row_sha256"
        }
        if not hmac.compare_digest(observed, canonical_sha256(body)):
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "target row self-hash changed"
            )
        targets.append(copy.deepcopy(row))
    if targets[0]["target_exposure"] != 1:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "genesis target must be LONG"
        )

    account = _Account(
        cash=INITIAL_CASH,
        shares=0.0,
        exposure=0,
        equity=INITIAL_CASH,
        running_peak=INITIAL_CASH,
    )
    previous_equity = INITIAL_CASH
    previous_hash = GENESIS_PREVIOUS_ROW_SHA256
    ledger_rows: list[dict[str, Any]] = []
    for ordinal, (market_row, target_row) in enumerate(
        zip(market, targets, strict=True),
        start=1,
    ):
        adjusted_open = _decode_float_hex(
            market_row["adjusted_open_hex"],
            f"ledger market row {ordinal}.adjusted_open_hex",
            positive=True,
        )
        prior_exposure = account.exposure
        account, diagnostics = _advance(
            account,
            adjusted_open=adjusted_open,
            target=target_row["target_exposure"],
            cost_bps=cost_bps,
        )
        period_factor = account.equity / previous_equity
        if not math.isfinite(period_factor) or period_factor <= 0.0:
            raise SecGemmaOnlineRiskOverlayLedgerError(
                "ledger period factor is invalid"
            )
        body = {
            "schema_version": LEDGER_ROW_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "policy_id": policy_id,
            "cost_bps": cost_bps,
            "ordinal": ordinal,
            "session": market_row["session"],
            "adjusted_open_hex": market_row["adjusted_open_hex"],
            "adjusted_close_hex": market_row["adjusted_close_hex"],
            "target_row_sha256": target_row["target_row_sha256"],
            "prior_exposure": prior_exposure,
            "target_exposure": account.exposure,
            "transition": diagnostics["transition"],
            "changing_leg": diagnostics["changing_leg"],
            "cash_hex": _float_hex(account.cash, "cash"),
            "shares_hex": _float_hex(account.shares, "shares"),
            "equity_before_fill_hex": _float_hex(
                diagnostics["equity_before_fill"],
                "equity before fill",
            ),
            "equity_open_hex": _float_hex(account.equity, "equity"),
            "period_factor_hex": _float_hex(period_factor, "period factor"),
            "drawdown_hex": _float_hex(
                diagnostics["drawdown"], "drawdown"
            ),
            "previous_row_sha256": previous_hash,
        }
        row = {**body, "ledger_row_sha256": canonical_sha256(body)}
        ledger_rows.append(row)
        previous_hash = row["ledger_row_sha256"]
        previous_equity = account.equity

    terminal_market = market[-1]
    terminal_close = _decode_float_hex(
        terminal_market["adjusted_close_hex"],
        "terminal adjusted close",
        positive=True,
    )
    terminal_close_equity = account.cash + account.shares * terminal_close
    if not math.isfinite(terminal_close_equity) or terminal_close_equity <= 0.0:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "terminal close equity is invalid"
        )
    state = {
        "last_session": terminal_market["session"],
        "exposure": account.exposure,
        "cash_hex": _float_hex(account.cash, "terminal cash"),
        "shares_hex": _float_hex(account.shares, "terminal shares"),
        "terminal_open_equity_hex": _float_hex(
            account.equity, "terminal open equity"
        ),
        "terminal_close_equity_hex": _float_hex(
            terminal_close_equity, "terminal close equity"
        ),
        "minimum_cash_hex": min(
            _decode_float_hex(row["cash_hex"], "ledger cash", nonnegative=True)
            for row in ledger_rows
        ).hex(),
        "minimum_shares_hex": min(
            _decode_float_hex(
                row["shares_hex"], "ledger shares", nonnegative=True
            )
            for row in ledger_rows
        ).hex(),
        "maximum_exposure": max(
            _binary(row["target_exposure"], "ledger exposure")
            for row in ledger_rows
        ),
        "ledger_row_count": len(ledger_rows),
        "ledger_tip_sha256": previous_hash,
    }
    body = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "policy_id": policy_id,
        "cost_bps": cost_bps,
        "initial_cash_hex": INITIAL_CASH.hex(),
        "ledger_rows": ledger_rows,
        "ledger_rows_sha256": canonical_sha256(ledger_rows),
        "terminal_state": state,
        "terminal_state_sha256": canonical_sha256(state),
    }
    return {**body, "ledger_sha256": canonical_sha256(body)}


def validate_binary_ledger(
    ledger: Mapping[str, Any],
    *,
    expected_ledger_sha256: str,
    market_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    policy_id: str,
    cost_bps: int,
) -> str:
    """Independently rebuild one ledger and require exact identity."""

    if not isinstance(ledger, Mapping):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "ledger must be a mapping"
        )
    rebuilt = run_binary_ledger(
        market_rows=market_rows,
        target_rows=target_rows,
        policy_id=policy_id,
        cost_bps=cost_bps,
    )
    if dict(ledger) != rebuilt:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "ledger differs from deterministic replay"
        )
    observed = _sha256(
        ledger.get("ledger_sha256"), "ledger.ledger_sha256"
    )
    if not hmac.compare_digest(
        observed, _sha256(expected_ledger_sha256, "expected ledger hash")
    ):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "ledger is not externally pinned"
        )
    return observed


def _normalized_account(exposure: int, adjusted_open: float) -> _Account:
    exposure = _binary(exposure, "normalized account exposure")
    if exposure == 1:
        return _Account(
            cash=0.0,
            shares=1.0 / adjusted_open,
            exposure=1,
            equity=1.0,
            running_peak=1.0,
        )
    return _Account(
        cash=1.0,
        shares=0.0,
        exposure=0,
        equity=1.0,
        running_peak=1.0,
    )


def build_mature_counterfactual_lesson(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    baseline_signals: Sequence[Mapping[str, Any]],
    decision_session: str,
    accession_number: str,
    feature_row_sha256: str,
    feature_fit_eligible: bool,
    as_of_session: str,
) -> dict[str, Any]:
    """Build the fixed 10-bps lesson only after open ``t+21`` exists."""

    market = _canonical_market_rows(market_rows)
    signals = _canonical_baseline_signals(market, baseline_signals)
    decision_session = _iso_date(decision_session, "decision_session")
    as_of_session = _iso_date(as_of_session, "as_of_session")
    if not isinstance(accession_number, str) or not accession_number:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "accession_number is invalid"
        )
    feature_hash = _sha256(feature_row_sha256, "feature_row_sha256")
    if type(feature_fit_eligible) is not bool:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "feature_fit_eligible must be Boolean"
        )
    sessions = [row["session"] for row in market]
    if decision_session not in sessions:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "decision session is outside the market prefix"
        )
    if as_of_session not in sessions:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "as_of_session is outside the market prefix"
        )
    decision_position = sessions.index(decision_session)
    maturity_position = decision_position + LABEL_MATURITY_OFFSET
    if maturity_position >= len(sessions):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "counterfactual lesson has not matured in this market prefix"
        )
    maturity_session = sessions[maturity_position]
    if as_of_session < maturity_session:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "counterfactual lesson was requested before maturity"
        )
    baseline_targets = baseline_open_targets(market, signals)
    decision_open = _decode_float_hex(
        market[decision_position]["adjusted_open_hex"],
        "decision adjusted open",
        positive=True,
    )
    baseline = _normalized_account(
        baseline_targets[decision_position], decision_open
    )
    overlay = _normalized_account(
        baseline_targets[decision_position], decision_open
    )
    for position in range(decision_position + 1, maturity_position + 1):
        adjusted_open = _decode_float_hex(
            market[position]["adjusted_open_hex"],
            f"counterfactual open {position}",
            positive=True,
        )
        baseline_target = baseline_targets[position]
        forced_target = (
            0 if position < maturity_position else baseline_target
        )
        baseline, _ = _advance(
            baseline,
            adjusted_open=adjusted_open,
            target=baseline_target,
            cost_bps=10,
        )
        overlay, _ = _advance(
            overlay,
            adjusted_open=adjusted_open,
            target=forced_target,
            cost_bps=10,
        )
    incremental_edge = math.log(overlay.equity / baseline.equity)
    if not math.isfinite(incremental_edge):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "counterfactual edge is non-finite"
        )
    train_eligible = feature_fit_eligible
    body = {
        "schema_version": COUNTERFACTUAL_LESSON_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "accession_number": accession_number,
        "decision_session": decision_session,
        "entry_session": sessions[decision_position + 1],
        "maturity_session": maturity_session,
        "horizon_sessions": HORIZON_SESSIONS,
        "label_maturity_offset": LABEL_MATURITY_OFFSET,
        "cost_bps": 10,
        "feature_row_sha256": feature_hash,
        "feature_fit_eligible": feature_fit_eligible,
        "train_eligible": train_eligible,
        "audit_only": not train_eligible,
        "baseline_terminal_wealth_hex": _float_hex(
            baseline.equity, "baseline counterfactual wealth"
        ),
        "overlay_terminal_wealth_hex": _float_hex(
            overlay.equity, "overlay counterfactual wealth"
        ),
        "incremental_log_edge_10bps_hex": _float_hex(
            incremental_edge, "counterfactual edge"
        ),
        "binary_overlay_win": (
            1 if incremental_edge > POSITIVE_EDGE_TOLERANCE else 0
        ),
        "baseline_signal_prefix_sha256": canonical_sha256(
            signals[:maturity_position]
        ),
        "market_prefix_sha256": canonical_sha256(
            market[: maturity_position + 1]
        ),
    }
    return {
        **body,
        "counterfactual_lesson_sha256": canonical_sha256(body),
    }


def validate_mature_counterfactual_lesson(
    lesson: Mapping[str, Any],
    *,
    expected_counterfactual_lesson_sha256: str,
    market_rows: Sequence[Mapping[str, Any]],
    baseline_signals: Sequence[Mapping[str, Any]],
    decision_session: str,
    accession_number: str,
    feature_row_sha256: str,
    feature_fit_eligible: bool,
    as_of_session: str,
) -> str:
    """Rebuild one mature lesson and require exact externally pinned identity."""

    if not isinstance(lesson, Mapping):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "counterfactual lesson must be a mapping"
        )
    rebuilt = build_mature_counterfactual_lesson(
        market_rows=market_rows,
        baseline_signals=baseline_signals,
        decision_session=decision_session,
        accession_number=accession_number,
        feature_row_sha256=feature_row_sha256,
        feature_fit_eligible=feature_fit_eligible,
        as_of_session=as_of_session,
    )
    if dict(lesson) != rebuilt:
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "counterfactual lesson differs from deterministic replay"
        )
    observed = _sha256(
        lesson.get("counterfactual_lesson_sha256"),
        "counterfactual lesson hash",
    )
    expected = _sha256(
        expected_counterfactual_lesson_sha256,
        "expected counterfactual lesson hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "counterfactual lesson is not externally pinned"
        )
    return observed


def compare_ledgers(
    strategy: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare exact same-session ledger paths at open and terminal close."""

    left = _mapping(strategy, "strategy ledger")
    right = _mapping(benchmark, "benchmark ledger")
    left_rows = left.get("ledger_rows")
    right_rows = right.get("ledger_rows")
    if not isinstance(left_rows, list) or not isinstance(right_rows, list):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "ledger rows must be lists"
        )
    if len(left_rows) != len(right_rows) or any(
        left_row.get("session") != right_row.get("session")
        or left_row.get("adjusted_open_hex")
        != right_row.get("adjusted_open_hex")
        or left_row.get("adjusted_close_hex")
        != right_row.get("adjusted_close_hex")
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    ):
        raise SecGemmaOnlineRiskOverlayLedgerError(
            "strategy and benchmark ledgers are not same-ledger aligned"
        )
    left_state = _mapping(left.get("terminal_state"), "strategy state")
    right_state = _mapping(right.get("terminal_state"), "benchmark state")
    left_open = _decode_float_hex(
        left_state.get("terminal_open_equity_hex"),
        "strategy terminal open equity",
        positive=True,
    )
    right_open = _decode_float_hex(
        right_state.get("terminal_open_equity_hex"),
        "benchmark terminal open equity",
        positive=True,
    )
    left_close = _decode_float_hex(
        left_state.get("terminal_close_equity_hex"),
        "strategy terminal close equity",
        positive=True,
    )
    right_close = _decode_float_hex(
        right_state.get("terminal_close_equity_hex"),
        "benchmark terminal close equity",
        positive=True,
    )
    body = {
        "strategy_ledger_sha256": _sha256(
            left.get("ledger_sha256"), "strategy ledger hash"
        ),
        "benchmark_ledger_sha256": _sha256(
            right.get("ledger_sha256"), "benchmark ledger hash"
        ),
        "adjusted_open_active_log_edge_hex": _float_hex(
            math.log(left_open / right_open), "open active edge"
        ),
        "terminal_adjusted_close_active_log_edge_hex": _float_hex(
            math.log(left_close / right_close), "close active edge"
        ),
        "strategy_terminal_open_equity_hex": left_open.hex(),
        "benchmark_terminal_open_equity_hex": right_open.hex(),
        "strategy_terminal_close_equity_hex": left_close.hex(),
        "benchmark_terminal_close_equity_hex": right_close.hex(),
    }
    return {**body, "ledger_comparison_sha256": canonical_sha256(body)}


__all__ = [
    "ALLOWED_COST_BPS",
    "COUNTERFACTUAL_LESSON_SCHEMA_VERSION",
    "GENESIS_PREVIOUS_ROW_SHA256",
    "INITIAL_CASH",
    "LEDGER_ROW_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "OVERLAY_EPISODE_OBSERVATION_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayLedgerError",
    "TARGET_STREAM_SCHEMA_VERSION",
    "TERMINAL_PENDING_STATE_SCHEMA_VERSION",
    "baseline_open_targets",
    "build_baseline_signal_row",
    "build_combined_target_rows",
    "build_mature_counterfactual_lesson",
    "compare_ledgers",
    "run_binary_ledger",
    "validate_binary_ledger",
    "validate_mature_counterfactual_lesson",
]
