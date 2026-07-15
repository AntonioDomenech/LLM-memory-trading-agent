"""Pure continuous binary LONG/CASH ledger for contextual aggregation.

The module is deliberately self-contained.  It performs no data acquisition,
network access, model inference, or reporting-window reset.  Callers supply an
already-authorized adjusted-open series and one exact close target per session.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date
from numbers import Real
from typing import Any, Mapping, Sequence

import pandas as pd


ACCOUNT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
ACCOUNT_INCEPTION_DATE = "2005-01-01"
ACCOUNT_INCEPTION_FILL_DATE = "2005-01-03"
INITIAL_CASH = 1000.0
RECONCILIATION_TOLERANCE = 1e-10
PRIMARY_CASH_COMPARATOR_LONG = "primary_cash_comparator_long"
PRIMARY_LONG_COMPARATOR_CASH = "primary_long_comparator_cash"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_POLICY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_GENESIS_TIP = "sha256:" + hashlib.sha256(
    b"contextual-expert-aggregation-ledger-genesis-v1"
).hexdigest()


class BinaryLedgerError(RuntimeError):
    """Raised when ledger, checkpoint, or attribution evidence is invalid."""


def _canonical_json_bytes(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return item
        if isinstance(item, Real):
            number = float(item)
            if not math.isfinite(number):
                raise BinaryLedgerError("Canonical evidence contains a nonfinite number")
            if isinstance(item, int) and not isinstance(item, bool):
                return int(item)
            return number
        raise BinaryLedgerError("Canonical evidence contains an unsupported value")

    return json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_json_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_keys(value: Any, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise BinaryLedgerError(f"{field} does not have the exact frozen schema")
    return value


def _finite_number(
    value: Any,
    *,
    field: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BinaryLedgerError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise BinaryLedgerError(f"{field} must be finite")
    if number != 0.0 and abs(number) < sys.float_info.min:
        raise BinaryLedgerError(f"{field} must not be subnormal")
    if positive and number <= 0.0:
        raise BinaryLedgerError(f"{field} must be positive")
    if nonnegative and number < 0.0:
        raise BinaryLedgerError(f"{field} must be nonnegative")
    return 0.0 if number == 0.0 else number


def _strict_float(
    value: Any,
    *,
    field: str,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    """Validate a canonical JSON floating-point field without numeric coercion."""

    if type(value) is not float:
        raise BinaryLedgerError(f"{field} must be a canonical float")
    return _finite_number(
        value,
        field=field,
        nonnegative=nonnegative,
        positive=positive,
    )


def _strict_schema_version(value: Any, *, field: str, expected: int) -> int:
    if type(value) is not int or value != expected:
        raise BinaryLedgerError(f"{field} changed")
    return value


def _binary(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BinaryLedgerError(f"{field} must be exact binary 0 or 1")
    number = _finite_number(value, field=field)
    if number not in (0.0, 1.0):
        raise BinaryLedgerError(f"{field} must be exact binary 0 or 1")
    return int(number)


def _strict_binary_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise BinaryLedgerError(f"{field} must be canonical integer 0 or 1")
    return value


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BinaryLedgerError(f"{field} must be an integer >= {minimum}")
    return value


def _policy(value: Any) -> str:
    if not isinstance(value, str) or _POLICY_RE.fullmatch(value) is None:
        raise BinaryLedgerError("policy_name is invalid")
    return value


def _cost_bps(value: Any) -> float:
    bps = _finite_number(value, field="cost_bps", nonnegative=True)
    if bps >= 10_000.0:
        raise BinaryLedgerError("cost_bps must be below 10000")
    return bps


def _strict_iso(value: Any, *, field: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str):
        raise BinaryLedgerError(f"{field} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise BinaryLedgerError(f"{field} must be an ISO date string") from exc
    if parsed.isoformat() != value:
        raise BinaryLedgerError(f"{field} must be a canonical ISO date")
    return value


def _input_iso(value: Any, *, field: str) -> str:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise BinaryLedgerError(f"{field} contains an invalid session date") from exc
    if stamp.tz is not None or stamp != stamp.normalize():
        raise BinaryLedgerError(f"{field} dates must be timezone-naive midnights")
    return stamp.date().isoformat()


def _optional_binary(value: Any, *, field: str) -> int | None:
    return None if value is None else _strict_binary_int(value, field=field)


@dataclass(frozen=True)
class AccountState:
    schema_version: int
    policy_name: str
    cost_bps: float
    inception_fill_date: str | None
    inception_count: int
    last_session_date: str | None
    last_reference_price: float | None
    last_trade_fill_date: str | None
    cash: float
    shares: float
    held_target: int | None
    previous_requested_target: int | None
    pending_decision_date: str | None
    pending_target_exposure: int | None
    last_equity: float
    running_peak: float
    cumulative_active_log_edge: float
    ledger_row_count: int
    ledger_tip_sha256: str
    open_cash_entry_decision_date: str | None
    open_cash_entry_fill_date: str | None
    open_cash_entry_reference_price: float | None
    open_cash_fill_observations: int

    def __post_init__(self) -> None:
        _validate_account_state(self)

    @classmethod
    def initial(cls, *, policy_name: str, cost_bps: float) -> "AccountState":
        return cls(
            schema_version=ACCOUNT_SCHEMA_VERSION,
            policy_name=_policy(policy_name),
            cost_bps=_cost_bps(cost_bps),
            inception_fill_date=None,
            inception_count=0,
            last_session_date=None,
            last_reference_price=None,
            last_trade_fill_date=None,
            cash=INITIAL_CASH,
            shares=0.0,
            held_target=None,
            previous_requested_target=None,
            pending_decision_date=None,
            pending_target_exposure=None,
            last_equity=INITIAL_CASH,
            running_peak=INITIAL_CASH,
            cumulative_active_log_edge=0.0,
            ledger_row_count=0,
            ledger_tip_sha256=_GENESIS_TIP,
            open_cash_entry_decision_date=None,
            open_cash_entry_fill_date=None,
            open_cash_entry_reference_price=None,
            open_cash_fill_observations=0,
        )

    def to_checkpoint(self) -> dict[str, Any]:
        state = asdict(self)
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "account_state": state,
        }
        return {**payload, "account_state_sha256": _sha256(payload)}

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> "AccountState":
        value = _strict_keys(
            checkpoint,
            {
                "checkpoint_schema_version",
                "account_state",
                "account_state_sha256",
            },
            field="account checkpoint",
        )
        _strict_schema_version(
            value["checkpoint_schema_version"],
            field="Account checkpoint schema version",
            expected=CHECKPOINT_SCHEMA_VERSION,
        )
        if not isinstance(value["account_state_sha256"], str) or _SHA256_RE.fullmatch(
            value["account_state_sha256"]
        ) is None:
            raise BinaryLedgerError("Account checkpoint self-hash is invalid")
        expected_hash = _sha256(
            {
                "checkpoint_schema_version": value["checkpoint_schema_version"],
                "account_state": value["account_state"],
            }
        )
        if value["account_state_sha256"] != expected_hash:
            raise BinaryLedgerError("Account checkpoint self-hash is invalid")
        state_value = _strict_keys(
            value["account_state"],
            set(AccountState.__dataclass_fields__),
            field="account_state",
        )
        return cls(**dict(state_value))


def _validate_account_state(state: AccountState) -> None:
    _strict_schema_version(
        state.schema_version,
        field="AccountState schema version",
        expected=ACCOUNT_SCHEMA_VERSION,
    )
    _policy(state.policy_name)
    _strict_float(state.cost_bps, field="cost_bps", nonnegative=True)
    if state.cost_bps >= 10_000.0:
        raise BinaryLedgerError("cost_bps must be below 10000")
    count = _strict_int(state.inception_count, field="inception_count")
    rows = _strict_int(state.ledger_row_count, field="ledger_row_count")
    cash = _strict_float(state.cash, field="cash", nonnegative=True)
    shares = _strict_float(state.shares, field="shares", nonnegative=True)
    equity = _strict_float(state.last_equity, field="last_equity", positive=True)
    peak = _strict_float(state.running_peak, field="running_peak", positive=True)
    _strict_float(
        state.cumulative_active_log_edge,
        field="cumulative_active_log_edge",
    )
    if peak < equity:
        raise BinaryLedgerError("running_peak is below last_equity")
    if not isinstance(state.ledger_tip_sha256, str) or _SHA256_RE.fullmatch(
        state.ledger_tip_sha256
    ) is None:
        raise BinaryLedgerError("ledger_tip_sha256 is invalid")

    inception = _strict_iso(
        state.inception_fill_date, field="inception_fill_date", optional=True
    )
    last_session = _strict_iso(
        state.last_session_date, field="last_session_date", optional=True
    )
    last_trade = _strict_iso(
        state.last_trade_fill_date, field="last_trade_fill_date", optional=True
    )
    pending_date = _strict_iso(
        state.pending_decision_date, field="pending_decision_date", optional=True
    )
    held = _optional_binary(state.held_target, field="held_target")
    previous = _optional_binary(
        state.previous_requested_target, field="previous_requested_target"
    )
    pending = _optional_binary(
        state.pending_target_exposure, field="pending_target_exposure"
    )
    open_decision = _strict_iso(
        state.open_cash_entry_decision_date,
        field="open_cash_entry_decision_date",
        optional=True,
    )
    open_fill = _strict_iso(
        state.open_cash_entry_fill_date,
        field="open_cash_entry_fill_date",
        optional=True,
    )
    open_observations = _strict_int(
        state.open_cash_fill_observations,
        field="open_cash_fill_observations",
    )

    if count == 0:
        if (
            inception is not None
            or last_session is not None
            or state.last_reference_price is not None
            or last_trade is not None
            or held is not None
            or previous is not None
            or pending_date is not None
            or pending is not None
            or open_decision is not None
            or open_fill is not None
            or state.open_cash_entry_reference_price is not None
            or open_observations != 0
            or cash != INITIAL_CASH
            or shares != 0.0
            or equity != INITIAL_CASH
            or peak != INITIAL_CASH
            or state.cumulative_active_log_edge != 0.0
            or rows != 0
            or state.ledger_tip_sha256 != _GENESIS_TIP
        ):
            raise BinaryLedgerError("Pre-inception AccountState is not exact")
        return

    if count != 1 or rows < 1:
        raise BinaryLedgerError("Account may be incepted exactly once")
    if inception != ACCOUNT_INCEPTION_FILL_DATE:
        raise BinaryLedgerError("Account inception fill date is not exact")
    if last_session is None or last_session < inception:
        raise BinaryLedgerError("Account last session is invalid")
    reference = _strict_float(
        state.last_reference_price, field="last_reference_price", positive=True
    )
    if last_trade is None or last_trade < inception or last_trade > last_session:
        raise BinaryLedgerError("last_trade_fill_date is invalid")
    if held is None or previous != held:
        raise BinaryLedgerError("Held and previous requested targets differ")
    if pending_date != last_session or pending is None:
        raise BinaryLedgerError("Cutoff-close target is not pending for next open")
    if held == 1:
        if cash != 0.0 or shares <= 0.0:
            raise BinaryLedgerError("LONG state must have shares and exact zero cash")
        if any(
            item is not None
            for item in (open_decision, open_fill, state.open_cash_entry_reference_price)
        ) or open_observations != 0:
            raise BinaryLedgerError("LONG state carries an impossible open CASH episode")
    else:
        if shares != 0.0 or cash <= 0.0:
            raise BinaryLedgerError("CASH state must have cash and exact zero shares")
        if open_decision is None or open_fill is None:
            raise BinaryLedgerError("CASH state omits its episode entry")
        _strict_float(
            state.open_cash_entry_reference_price,
            field="open_cash_entry_reference_price",
            positive=True,
        )
        if not (open_decision < open_fill <= last_session):
            raise BinaryLedgerError("Open CASH episode dates are inconsistent")
        if open_observations < 1:
            raise BinaryLedgerError("Open CASH episode has no held-CASH observations")
    if reference <= 0.0:
        raise BinaryLedgerError("last_reference_price is invalid")
    expected_equity = cash + shares * reference
    if expected_equity != equity:
        raise BinaryLedgerError("AccountState equity does not match cash and shares")


LEDGER_COLUMNS = (
    "policy_name",
    "cost_bps",
    "row_index",
    "decision_date",
    "fill_date",
    "reference_adjusted_open",
    "fill_price",
    "transition",
    "inception_fill",
    "target_changed",
    "trade_executed",
    "prior_requested_target_exposure",
    "held_exposure_before_fill",
    "requested_target_exposure",
    "post_fill_exposure",
    "close_decision_target_exposure",
    "cash_before_fill",
    "shares_before_fill",
    "equity_before_fill",
    "signed_share_delta",
    "cash_after_fill",
    "shares_after_fill",
    "equity",
    "turnover_reference",
    "slippage",
    "fees",
    "margin_interest",
    "cash_interest",
    "daily_return",
    "drawdown",
    "active_log_increment_vs_buy_hold",
    "cumulative_active_log_edge",
    "previous_row_sha256",
    "row_sha256",
)

_LEDGER_FLOAT_POSITIVE = {
    "reference_adjusted_open",
    "fill_price",
    "equity_before_fill",
    "equity",
}
_LEDGER_FLOAT_NONNEGATIVE = {
    "cash_before_fill",
    "shares_before_fill",
    "cash_after_fill",
    "shares_after_fill",
    "turnover_reference",
    "slippage",
    "fees",
    "margin_interest",
    "cash_interest",
}
_LEDGER_FLOAT_SIGNED = {
    "signed_share_delta",
    "daily_return",
    "drawdown",
    "active_log_increment_vs_buy_hold",
    "cumulative_active_log_edge",
}
_LEDGER_BINARY_COLUMNS = {
    "held_exposure_before_fill",
    "requested_target_exposure",
    "post_fill_exposure",
    "close_decision_target_exposure",
}
_LEDGER_BOOLEAN_COLUMNS = {"inception_fill", "target_changed", "trade_executed"}
_TRANSITIONS = {"INCEPTION", "BUY", "SELL", "HOLD_LONG", "HOLD_CASH"}


def _validate_observed_ledger_row(row: Mapping[str, Any]) -> None:
    if set(row) != set(LEDGER_COLUMNS):
        raise BinaryLedgerError("Ledger row does not have the exact canonical schema")
    _policy(row["policy_name"])
    cost = _strict_float(row["cost_bps"], field="ledger cost_bps", nonnegative=True)
    if cost >= 10_000.0:
        raise BinaryLedgerError("ledger cost_bps must be below 10000")
    _strict_int(row["row_index"], field="row_index", minimum=1)
    if type(row["decision_date"]) is not str:
        raise BinaryLedgerError("decision_date must be a canonical string")
    _strict_iso(row["fill_date"], field="fill_date")
    if type(row["transition"]) is not str or row["transition"] not in _TRANSITIONS:
        raise BinaryLedgerError("transition is invalid")
    for name in _LEDGER_FLOAT_POSITIVE:
        _strict_float(row[name], field=name, positive=True)
    for name in _LEDGER_FLOAT_NONNEGATIVE:
        _strict_float(row[name], field=name, nonnegative=True)
    for name in _LEDGER_FLOAT_SIGNED:
        _strict_float(row[name], field=name)
    for name in _LEDGER_BINARY_COLUMNS:
        _strict_binary_int(row[name], field=name)
    for name in _LEDGER_BOOLEAN_COLUMNS:
        if type(row[name]) is not bool:
            raise BinaryLedgerError(f"{name} must be a canonical Boolean")
    prior = row["prior_requested_target_exposure"]
    if prior != "":
        _strict_binary_int(prior, field="prior_requested_target_exposure")
    for name in ("previous_row_sha256", "row_sha256"):
        if type(row[name]) is not str or _SHA256_RE.fullmatch(row[name]) is None:
            raise BinaryLedgerError(f"{name} is invalid")
    expected_hash = _sha256({name: row[name] for name in LEDGER_COLUMNS[:-1]})
    if row["row_sha256"] != expected_hash:
        raise BinaryLedgerError("Ledger row self-hash is invalid")


def _exact_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    columns: Sequence[str],
    field: str,
) -> None:
    if not isinstance(observed, pd.DataFrame) or tuple(observed.columns) != tuple(columns):
        raise BinaryLedgerError(f"{field} does not have the exact canonical columns")
    observed_payload = {
        "columns": list(columns),
        "rows": observed.to_dict(orient="records"),
    }
    expected_payload = {
        "columns": list(columns),
        "rows": expected.to_dict(orient="records"),
    }
    if _canonical_json_bytes(observed_payload) != _canonical_json_bytes(expected_payload):
        raise BinaryLedgerError(f"{field} differs from deterministic regeneration")


@dataclass(frozen=True)
class LedgerRun:
    ledger: pd.DataFrame
    state: AccountState


def _transition_cost(before: int, after: int, cost: float) -> float:
    if before == after:
        return 0.0
    if before == 1 and after == 0:
        return math.log1p(-cost)
    if before == 0 and after == 1:
        return -math.log1p(cost)
    raise BinaryLedgerError("Transition is not binary")


def _advance_one(
    state: AccountState,
    *,
    fill_date: str,
    reference_price: float,
    close_target: int,
) -> tuple[dict[str, Any], AccountState]:
    ref = _finite_number(reference_price, field="reference_adjusted_open", positive=True)
    close_value = _binary(close_target, field="close target")
    cost = state.cost_bps / 10_000.0
    inception = state.inception_count == 0
    if inception:
        if fill_date != ACCOUNT_INCEPTION_FILL_DATE:
            raise BinaryLedgerError("Initial purchase must use the exact inception fill")
        decision_date = ""
        requested = 1
        held_before = 0
        prior: int | str = ""
    else:
        if state.last_session_date is None or fill_date <= state.last_session_date:
            raise BinaryLedgerError("Ledger sessions are not strict continuations")
        if (
            state.pending_decision_date is None
            or state.pending_target_exposure is None
            or state.pending_decision_date >= fill_date
        ):
            raise BinaryLedgerError("No causal prior-close target is pending")
        decision_date = state.pending_decision_date
        requested = state.pending_target_exposure
        held_before = int(state.held_target)
        prior = int(state.previous_requested_target)

    cash_before = state.cash
    shares_before = state.shares
    equity_before = _finite_number(
        cash_before + shares_before * ref,
        field="equity_before_fill",
        positive=True,
    )
    target_changed = inception or requested != held_before
    delta = 0.0
    fill_price = ref
    slippage = 0.0
    cash_after = cash_before
    shares_after = shares_before
    if target_changed:
        if requested == 1:
            if shares_before != 0.0 or cash_before <= 0.0:
                raise BinaryLedgerError("Binary BUY requires an all-CASH account")
            fill_price = ref * (1.0 + cost)
            delta = cash_before / fill_price
            shares_after = _finite_number(delta, field="bought shares", positive=True)
            cash_after = 0.0
            slippage = shares_after * (fill_price - ref)
            transition = "INCEPTION" if inception else "BUY"
        else:
            if cash_before != 0.0 or shares_before <= 0.0:
                raise BinaryLedgerError("Binary SELL requires an all-LONG account")
            fill_price = ref * (1.0 - cost)
            delta = -shares_before
            cash_after = shares_before * fill_price
            shares_after = 0.0
            slippage = shares_before * (ref - fill_price)
            transition = "SELL"
    else:
        transition = "HOLD_LONG" if requested == 1 else "HOLD_CASH"

    cash_after = _finite_number(cash_after, field="cash_after_fill", nonnegative=True)
    shares_after = _finite_number(
        shares_after, field="shares_after_fill", nonnegative=True
    )
    equity_after = _finite_number(
        cash_after + shares_after * ref, field="equity", positive=True
    )
    if requested == 1 and (cash_after != 0.0 or shares_after <= 0.0):
        raise BinaryLedgerError("LONG fill is not exactly all invested")
    if requested == 0 and (shares_after != 0.0 or cash_after <= 0.0):
        raise BinaryLedgerError("CASH fill is not exactly all cash")

    if inception:
        active_increment = 0.0
    else:
        assert state.last_reference_price is not None
        active_increment = (
            (held_before - 1) * math.log(ref / state.last_reference_price)
            + _transition_cost(held_before, requested, cost)
        )
    cumulative = state.cumulative_active_log_edge + active_increment
    daily_return = equity_after / state.last_equity - 1.0
    running_peak = max(state.running_peak, equity_after)
    drawdown = equity_after / running_peak - 1.0
    turnover = abs(delta) * ref / equity_before

    open_decision = state.open_cash_entry_decision_date
    open_fill = state.open_cash_entry_fill_date
    open_ref = state.open_cash_entry_reference_price
    open_observations = state.open_cash_fill_observations
    if transition == "SELL":
        open_decision = decision_date
        open_fill = fill_date
        open_ref = ref
        open_observations = 1
    elif transition == "HOLD_CASH":
        open_observations += 1
    elif transition in {"BUY", "INCEPTION"}:
        open_decision = None
        open_fill = None
        open_ref = None
        open_observations = 0

    row_without_hash: dict[str, Any] = {
        "policy_name": state.policy_name,
        "cost_bps": state.cost_bps,
        "row_index": state.ledger_row_count + 1,
        "decision_date": decision_date,
        "fill_date": fill_date,
        "reference_adjusted_open": ref,
        "fill_price": fill_price,
        "transition": transition,
        "inception_fill": inception,
        "target_changed": target_changed,
        "trade_executed": target_changed,
        "prior_requested_target_exposure": prior,
        "held_exposure_before_fill": held_before,
        "requested_target_exposure": requested,
        "post_fill_exposure": requested,
        "close_decision_target_exposure": close_value,
        "cash_before_fill": cash_before,
        "shares_before_fill": shares_before,
        "equity_before_fill": equity_before,
        "signed_share_delta": delta,
        "cash_after_fill": cash_after,
        "shares_after_fill": shares_after,
        "equity": equity_after,
        "turnover_reference": turnover,
        "slippage": slippage,
        "fees": 0.0,
        "margin_interest": 0.0,
        "cash_interest": 0.0,
        "daily_return": daily_return,
        "drawdown": drawdown,
        "active_log_increment_vs_buy_hold": active_increment,
        "cumulative_active_log_edge": cumulative,
        "previous_row_sha256": state.ledger_tip_sha256,
    }
    row = {**row_without_hash, "row_sha256": _sha256(row_without_hash)}
    new_state = AccountState(
        schema_version=ACCOUNT_SCHEMA_VERSION,
        policy_name=state.policy_name,
        cost_bps=state.cost_bps,
        inception_fill_date=(fill_date if inception else state.inception_fill_date),
        inception_count=1,
        last_session_date=fill_date,
        last_reference_price=ref,
        last_trade_fill_date=(
            fill_date if target_changed else state.last_trade_fill_date
        ),
        cash=cash_after,
        shares=shares_after,
        held_target=requested,
        previous_requested_target=requested,
        pending_decision_date=fill_date,
        pending_target_exposure=close_value,
        last_equity=equity_after,
        running_peak=running_peak,
        cumulative_active_log_edge=cumulative,
        ledger_row_count=state.ledger_row_count + 1,
        ledger_tip_sha256=row["row_sha256"],
        open_cash_entry_decision_date=open_decision,
        open_cash_entry_fill_date=open_fill,
        open_cash_entry_reference_price=open_ref,
        open_cash_fill_observations=open_observations,
    )
    return row, new_state


def _as_series(value: Any, *, field: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.copy()
    if isinstance(value, Mapping):
        return pd.Series(dict(value))
    raise BinaryLedgerError(f"{field} must be a pandas Series or mapping")


def _canonical_inputs(
    adjusted_opens: Any, close_targets: Any
) -> list[tuple[str, float, int]]:
    opens = _as_series(adjusted_opens, field="adjusted_opens")
    targets = _as_series(close_targets, field="close_targets")
    open_dates = [_input_iso(value, field="adjusted_opens") for value in opens.index]
    target_dates = [_input_iso(value, field="close_targets") for value in targets.index]
    if (
        open_dates != target_dates
        or len(set(open_dates)) != len(open_dates)
        or any(right <= left for left, right in zip(open_dates, open_dates[1:]))
    ):
        raise BinaryLedgerError("Prices and targets require one identical strict date sequence")
    result: list[tuple[str, float, int]] = []
    for day, price, target in zip(open_dates, opens.tolist(), targets.tolist()):
        result.append(
            (
                day,
                _finite_number(price, field="adjusted open", positive=True),
                _binary(target, field="close target"),
            )
        )
    return result


def _run_canonical_inputs(
    inputs: Sequence[tuple[str, float, int]], *, state: AccountState
) -> LedgerRun:
    if state.inception_count == 0 and not any(
        day == ACCOUNT_INCEPTION_FILL_DATE for day, _, _ in inputs
    ):
        raise BinaryLedgerError("Genesis input omits the exact inception fill session")
    if state.last_session_date is not None and inputs:
        if inputs[0][0] <= state.last_session_date:
            raise BinaryLedgerError("Continuation overlaps or replays an earlier session")
    rows: list[dict[str, Any]] = []
    for fill_date, ref, close_target in inputs:
        if state.inception_count == 0 and fill_date < ACCOUNT_INCEPTION_FILL_DATE:
            continue
        row, state = _advance_one(
            state,
            fill_date=fill_date,
            reference_price=ref,
            close_target=close_target,
        )
        rows.append(row)
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    return LedgerRun(ledger=ledger, state=state)


def run_continuous_ledger(
    adjusted_opens: Any,
    close_targets: Any,
    *,
    policy_name: str,
    cost_bps: float,
) -> LedgerRun:
    """Run the one genesis account from the exact 2005-01-03 initial fill."""

    policy = _policy(policy_name)
    cost = _cost_bps(cost_bps)
    inputs = _canonical_inputs(adjusted_opens, close_targets)
    return _run_canonical_inputs(
        inputs,
        state=AccountState.initial(policy_name=policy, cost_bps=cost),
    )


def verify_continuation_checkpoint(
    prior_adjusted_opens: Any,
    prior_close_targets: Any,
    prior_ledger: pd.DataFrame,
    checkpoint: Mapping[str, Any],
    *,
    policy_name: str,
    cost_bps: float,
) -> AccountState:
    """Replay an exact prefix from genesis and return its verified state.

    A checkpoint's self-hash proves only that its bytes are internally
    consistent.  This verifier additionally proves that the state is the exact
    result of the supplied inputs and canonical ledger from the fixed initial
    account.  Continuation execution calls this verifier internally; callers
    cannot feed the returned state into a suffix-only execution path.
    """

    policy = _policy(policy_name)
    cost = _cost_bps(cost_bps)
    inputs = _canonical_inputs(prior_adjusted_opens, prior_close_targets)
    canonical_index = pd.DatetimeIndex([item[0] for item in inputs])
    canonical_opens = pd.Series(
        [item[1] for item in inputs], index=canonical_index, dtype=float
    )
    canonical_targets = pd.Series(
        [item[2] for item in inputs], index=canonical_index, dtype=int
    )
    replay = run_continuous_ledger(
        canonical_opens,
        canonical_targets,
        policy_name=policy,
        cost_bps=cost,
    )
    restored = AccountState.from_checkpoint(checkpoint)
    if restored.inception_count != 1:
        raise BinaryLedgerError(
            "Continuation checkpoint must follow an incepted ledger prefix"
        )
    if restored != replay.state:
        raise BinaryLedgerError(
            "Continuation checkpoint differs from full prefix replay"
        )
    _exact_frame_equal(
        prior_ledger,
        replay.ledger,
        columns=LEDGER_COLUMNS,
        field="Prior ledger",
    )
    verify_ledger(
        prior_ledger,
        start_state=AccountState.initial(policy_name=policy, cost_bps=cost),
        expected_end_state=restored,
    )
    return restored


def run_verified_continuation(
    prior_adjusted_opens: Any,
    prior_close_targets: Any,
    prior_ledger: pd.DataFrame,
    checkpoint: Mapping[str, Any],
    suffix_adjusted_opens: Any,
    suffix_close_targets: Any,
    *,
    policy_name: str,
    cost_bps: float,
) -> LedgerRun:
    """Verify full prefix truth before reading or executing any suffix value."""

    policy = _policy(policy_name)
    cost = _cost_bps(cost_bps)
    state = verify_continuation_checkpoint(
        prior_adjusted_opens,
        prior_close_targets,
        prior_ledger,
        checkpoint,
        policy_name=policy,
        cost_bps=cost,
    )
    # This is intentionally the first access to either suffix object.
    suffix_inputs = _canonical_inputs(suffix_adjusted_opens, suffix_close_targets)
    return _run_canonical_inputs(suffix_inputs, state=state)


def verify_ledger(
    ledger: pd.DataFrame,
    *,
    start_state: AccountState,
    expected_end_state: AccountState | None = None,
) -> AccountState:
    """Replay every canonical row and verify semantics, arithmetic, and hashes."""

    if not isinstance(ledger, pd.DataFrame) or tuple(ledger.columns) != LEDGER_COLUMNS:
        raise BinaryLedgerError("Ledger does not have the exact canonical columns")
    if not isinstance(start_state, AccountState):
        raise BinaryLedgerError("start_state must be an AccountState")
    state = start_state
    for observed in ledger.to_dict(orient="records"):
        _validate_observed_ledger_row(observed)
        fill_date = _strict_iso(observed["fill_date"], field="fill_date")
        ref = _finite_number(
            observed["reference_adjusted_open"],
            field="reference_adjusted_open",
            positive=True,
        )
        close_target = _binary(
            observed["close_decision_target_exposure"],
            field="close_decision_target_exposure",
        )
        expected, state = _advance_one(
            state,
            fill_date=str(fill_date),
            reference_price=ref,
            close_target=close_target,
        )
        if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
            raise BinaryLedgerError("Ledger row differs from deterministic replay")
    if expected_end_state is not None and state != expected_end_state:
        raise BinaryLedgerError("Ledger replay does not reach its declared end state")
    return state


def assert_no_leverage(
    ledger: pd.DataFrame, *, start_state: AccountState | None = None
) -> dict[str, Any]:
    if not isinstance(ledger, pd.DataFrame) or tuple(ledger.columns) != LEDGER_COLUMNS:
        raise BinaryLedgerError("No-leverage proof requires a canonical ledger")
    if ledger.empty:
        raise BinaryLedgerError("No-leverage proof requires an incepted ledger")
    first = ledger.to_dict(orient="records")[0]
    _validate_observed_ledger_row(first)
    if start_state is None:
        if first["inception_fill"] is not True:
            raise BinaryLedgerError("Suffix no-leverage proof requires its start_state")
        start_state = AccountState.initial(
            policy_name=first["policy_name"], cost_bps=first["cost_bps"]
        )
    verify_ledger(ledger, start_state=start_state)
    max_exposure = 0
    for row in ledger.to_dict(orient="records"):
        requested = _strict_binary_int(
            row["requested_target_exposure"], field="requested target"
        )
        held = _strict_binary_int(
            row["held_exposure_before_fill"], field="held exposure"
        )
        post = _strict_binary_int(
            row["post_fill_exposure"], field="post-fill exposure"
        )
        close_target = _strict_binary_int(
            row["close_decision_target_exposure"], field="close target"
        )
        cash_before = _strict_float(
            row["cash_before_fill"], field="cash_before_fill", nonnegative=True
        )
        shares_before = _strict_float(
            row["shares_before_fill"], field="shares_before_fill", nonnegative=True
        )
        cash = _strict_float(
            row["cash_after_fill"], field="cash_after_fill", nonnegative=True
        )
        shares = _strict_float(
            row["shares_after_fill"], field="shares_after_fill", nonnegative=True
        )
        if requested != post or any(
            _strict_float(row[name], field=name) != 0.0
            for name in ("fees", "margin_interest", "cash_interest")
        ):
            raise BinaryLedgerError("Ledger violates the zero-financing binary contract")
        if held == 1 and (cash_before != 0.0 or shares_before <= 0.0):
            raise BinaryLedgerError("Pre-fill LONG state is leveraged or not all invested")
        if held == 0 and (shares_before != 0.0 or cash_before <= 0.0):
            raise BinaryLedgerError("Pre-fill CASH state has shares or nonpositive cash")
        if post == 1 and (cash != 0.0 or shares <= 0.0):
            raise BinaryLedgerError("LONG row is leveraged or not all invested")
        if post == 0 and (shares != 0.0 or cash <= 0.0):
            raise BinaryLedgerError("CASH row has shares or nonpositive cash")
        max_exposure = max(max_exposure, held, requested, post, close_target)
    return {
        "passed": True,
        "binary_requested_and_held": True,
        "nonnegative_cash_and_shares": True,
        "zero_margin_borrow_short_and_interest": True,
        "maximum_gross_exposure": max_exposure,
    }


_ACTION_COLUMNS = (
    "row_index",
    "decision_date",
    "fill_date",
    "transition",
    "inception_fill",
    "target_changed",
    "trade_executed",
    "prior_requested_target_exposure",
    "held_exposure_before_fill",
    "requested_target_exposure",
    "post_fill_exposure",
    "close_decision_target_exposure",
)

_CROSS_COST_INPUT_COLUMNS = (
    "row_index",
    "decision_date",
    "fill_date",
    "reference_adjusted_open",
    "close_decision_target_exposure",
)


def assert_cross_cost_action_identity(
    ledgers: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    if not isinstance(ledgers, Mapping) or len(ledgers) < 2:
        raise BinaryLedgerError("Cross-cost proof requires at least two ledgers")
    canonical: list[dict[str, Any]] | None = None
    canonical_input: list[dict[str, Any]] | None = None
    canonical_policy: str | None = None
    observed_cost_bps: list[float] = []
    for name in sorted(ledgers):
        frame = ledgers[name]
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != LEDGER_COLUMNS:
            raise BinaryLedgerError("Cross-cost action proof received a noncanonical ledger")
        records = frame.to_dict(orient="records")
        if not records:
            raise BinaryLedgerError("Cross-cost proof requires incepted ledgers")
        first = records[0]
        _validate_observed_ledger_row(first)
        policy = first["policy_name"]
        cost = first["cost_bps"]
        verify_ledger(
            frame,
            start_state=AccountState.initial(
                policy_name=policy,
                cost_bps=cost,
            ),
        )
        observed_cost_bps.append(cost)
        actions = frame.loc[:, list(_ACTION_COLUMNS)].to_dict(orient="records")
        input_stream = frame.loc[
            :, list(_CROSS_COST_INPUT_COLUMNS)
        ].to_dict(orient="records")
        if canonical is None:
            canonical = actions
            canonical_input = input_stream
            canonical_policy = policy
        elif policy != canonical_policy:
            raise BinaryLedgerError("Policy changed across costs")
        elif input_stream != canonical_input:
            raise BinaryLedgerError("Exogenous input stream changed across costs")
        elif actions != canonical:
            raise BinaryLedgerError("Action dates or targets changed across costs")
    assert canonical is not None
    assert canonical_input is not None
    if len(set(observed_cost_bps)) != len(observed_cost_bps):
        raise BinaryLedgerError("Cross-cost proof requires distinct cost_bps scenarios")
    sorted_cost_bps = sorted(observed_cost_bps)
    return {
        "passed": True,
        "cost_scenario_count": len(sorted_cost_bps),
        "observed_cost_bps": sorted_cost_bps,
        "action_stream_sha256": _sha256(canonical),
        "shared_input_stream_sha256": _sha256(canonical_input),
    }


def assert_always_long_matches_buy_hold(
    always_long: pd.DataFrame, buy_hold: pd.DataFrame
) -> dict[str, Any]:
    for frame in (always_long, buy_hold):
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != LEDGER_COLUMNS:
            raise BinaryLedgerError("Always-LONG proof requires canonical ledgers")
        if frame.empty:
            raise BinaryLedgerError("Always-LONG proof requires incepted ledgers")
        first = frame.to_dict(orient="records")[0]
        _validate_observed_ledger_row(first)
        end_state = verify_ledger(
            frame,
            start_state=AccountState.initial(
                policy_name=first["policy_name"], cost_bps=first["cost_bps"]
            ),
        )
        requested = frame["requested_target_exposure"].map(
            lambda value: _strict_binary_int(value, field="always-long target")
        )
        close_targets = frame["close_decision_target_exposure"].map(
            lambda value: _strict_binary_int(value, field="always-long close target")
        )
        if not (requested.eq(1).all() and close_targets.eq(1).all()) or (
            end_state.held_target != 1 or end_state.pending_target_exposure != 1
        ):
            raise BinaryLedgerError("Always-LONG or buy-hold target is not always LONG")
    ignored = {"policy_name", "previous_row_sha256", "row_sha256"}
    compared = [column for column in LEDGER_COLUMNS if column not in ignored]
    if not always_long.loc[:, compared].equals(buy_hold.loc[:, compared]):
        raise BinaryLedgerError("Always-LONG differs from same-ledger buy-and-hold")
    return {
        "passed": True,
        "rows": len(always_long),
        "economic_ledger_sha256": _sha256(
            always_long.loc[:, compared].to_dict(orient="records")
        ),
    }


EPISODE_COLUMNS = (
    "episode_id",
    "entry_decision_date",
    "entry_fill_date",
    "exit_decision_date",
    "exit_fill_date",
    "entry_reference_price",
    "entry_sell_fill_price",
    "exit_reference_price",
    "exit_buy_fill_price",
    "cash_fill_observations",
    "raw_active_log_edge",
    "cost_log_edge",
    "net_active_log_edge",
    "episode_sha256",
)

UNRESOLVED_EPISODE_COLUMNS = (
    "status",
    "entry_decision_date",
    "entry_fill_date",
    "entry_reference_price",
    "pending_decision_date",
    "pending_target_exposure",
    "mark_date",
    "mark_reference_price",
    "raw_active_log_edge_to_mark",
    "executed_cost_log_edge",
    "net_active_log_edge_to_mark",
    "unresolved_sha256",
)


@dataclass(frozen=True)
class EpisodeExtraction:
    complete: pd.DataFrame
    unresolved: pd.DataFrame


def extract_cash_episodes(
    ledger: pd.DataFrame,
    *,
    start_state: AccountState,
    expected_end_state: AccountState | None = None,
) -> EpisodeExtraction:
    end_state = verify_ledger(
        ledger, start_state=start_state, expected_end_state=expected_end_state
    )
    cost = start_state.cost_bps / 10_000.0
    entry_decision = start_state.open_cash_entry_decision_date
    entry_fill = start_state.open_cash_entry_fill_date
    entry_ref = start_state.open_cash_entry_reference_price
    cash_observations = start_state.open_cash_fill_observations
    complete: list[dict[str, Any]] = []
    for row in ledger.to_dict(orient="records"):
        if row["transition"] == "SELL":
            if entry_fill is not None:
                raise BinaryLedgerError("A CASH episode was opened twice")
            entry_decision = row["decision_date"]
            entry_fill = row["fill_date"]
            entry_ref = row["reference_adjusted_open"]
            cash_observations = 1
        elif row["requested_target_exposure"] == 0 and entry_fill is not None:
            cash_observations += 1
        if row["transition"] == "BUY":
            if entry_fill is None or entry_ref is None or entry_decision is None:
                raise BinaryLedgerError("A CASH episode exit lacks its entry")
            exit_ref = float(row["reference_adjusted_open"])
            raw = math.log(float(entry_ref) / exit_ref)
            cost_edge = math.log((1.0 - cost) / (1.0 + cost))
            payload: dict[str, Any] = {
                "episode_id": _sha256(
                    {
                        "policy": start_state.policy_name,
                        "entry_fill_date": entry_fill,
                    }
                ),
                "entry_decision_date": entry_decision,
                "entry_fill_date": entry_fill,
                "exit_decision_date": row["decision_date"],
                "exit_fill_date": row["fill_date"],
                "entry_reference_price": float(entry_ref),
                "entry_sell_fill_price": float(entry_ref) * (1.0 - cost),
                "exit_reference_price": exit_ref,
                "exit_buy_fill_price": exit_ref * (1.0 + cost),
                "cash_fill_observations": cash_observations,
                "raw_active_log_edge": raw,
                "cost_log_edge": cost_edge,
                "net_active_log_edge": raw + cost_edge,
            }
            complete.append({**payload, "episode_sha256": _sha256(payload)})
            entry_decision = None
            entry_fill = None
            entry_ref = None
            cash_observations = 0

    unresolved: list[dict[str, Any]] = []
    if end_state.held_target == 0:
        if entry_fill is None or entry_ref is None or ledger.empty:
            raise BinaryLedgerError("Open CASH tail lacks executable entry evidence")
        mark_ref = float(end_state.last_reference_price)
        raw = math.log(float(entry_ref) / mark_ref)
        cost_edge = math.log1p(-cost)
        payload = {
            "status": (
                "open_cash_pending_exit"
                if end_state.pending_target_exposure == 1
                else "open_cash"
            ),
            "entry_decision_date": entry_decision,
            "entry_fill_date": entry_fill,
            "entry_reference_price": float(entry_ref),
            "pending_decision_date": end_state.pending_decision_date,
            "pending_target_exposure": end_state.pending_target_exposure,
            "mark_date": end_state.last_session_date,
            "mark_reference_price": mark_ref,
            "raw_active_log_edge_to_mark": raw,
            "executed_cost_log_edge": cost_edge,
            "net_active_log_edge_to_mark": raw + cost_edge,
        }
        unresolved.append({**payload, "unresolved_sha256": _sha256(payload)})
    elif end_state.pending_target_exposure == 0:
        payload = {
            "status": "pending_cash_entry_unexecuted",
            "entry_decision_date": end_state.pending_decision_date,
            "entry_fill_date": "",
            "entry_reference_price": 0.0,
            "pending_decision_date": end_state.pending_decision_date,
            "pending_target_exposure": 0,
            "mark_date": end_state.last_session_date,
            "mark_reference_price": float(end_state.last_reference_price),
            "raw_active_log_edge_to_mark": 0.0,
            "executed_cost_log_edge": 0.0,
            "net_active_log_edge_to_mark": 0.0,
        }
        unresolved.append({**payload, "unresolved_sha256": _sha256(payload)})
    return EpisodeExtraction(
        complete=pd.DataFrame(complete, columns=EPISODE_COLUMNS),
        unresolved=pd.DataFrame(unresolved, columns=UNRESOLVED_EPISODE_COLUMNS),
    )


def _aligned_ledgers(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for frame in (left, right):
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != LEDGER_COLUMNS:
            raise BinaryLedgerError("Reconciliation requires canonical ledgers")
    fields = ["row_index", "fill_date", "reference_adjusted_open", "cost_bps"]
    if not left.loc[:, fields].equals(right.loc[:, fields]):
        raise BinaryLedgerError("Ledgers do not share exact fills, prices, and costs")


def reconcile_complete_cash_episodes(
    strategy: pd.DataFrame,
    buy_hold: pd.DataFrame,
    episodes: EpisodeExtraction,
    *,
    strategy_start_state: AccountState,
    buy_hold_start_state: AccountState,
    tolerance: float = RECONCILIATION_TOLERANCE,
) -> dict[str, Any]:
    tolerance_value = _finite_number(
        tolerance, field="reconciliation tolerance", positive=True
    )
    strategy_end = verify_ledger(strategy, start_state=strategy_start_state)
    benchmark_end = verify_ledger(buy_hold, start_state=buy_hold_start_state)
    _aligned_ledgers(strategy, buy_hold)
    if buy_hold_start_state.inception_count == 1 and (
        buy_hold_start_state.held_target != 1
        or buy_hold_start_state.pending_target_exposure != 1
    ):
        raise BinaryLedgerError("Same-ledger buy-and-hold starts outside LONG")
    benchmark_records = buy_hold.to_dict(orient="records")
    if not benchmark_records or any(
        _strict_binary_int(
            row["requested_target_exposure"],
            field="buy-and-hold requested target",
        )
        != 1
        or _strict_binary_int(
            row["close_decision_target_exposure"],
            field="buy-and-hold close target",
        )
        != 1
        for row in benchmark_records
    ):
        raise BinaryLedgerError(
            "Same-ledger buy-and-hold requested or queued a CASH target"
        )
    if benchmark_end.held_target != 1 or benchmark_end.pending_target_exposure != 1:
        raise BinaryLedgerError("Same-ledger buy-and-hold does not end LONG")
    if not isinstance(episodes, EpisodeExtraction):
        raise BinaryLedgerError("CASH episode evidence has the wrong type")
    regenerated = extract_cash_episodes(
        strategy,
        start_state=strategy_start_state,
        expected_end_state=strategy_end,
    )
    _exact_frame_equal(
        episodes.complete,
        regenerated.complete,
        columns=EPISODE_COLUMNS,
        field="Complete CASH episode evidence",
    )
    _exact_frame_equal(
        episodes.unresolved,
        regenerated.unresolved,
        columns=UNRESOLVED_EPISODE_COLUMNS,
        field="Unresolved CASH episode evidence",
    )
    episodes = regenerated
    if strategy_start_state.held_target == 0:
        raise BinaryLedgerError("Partially executed left-boundary CASH episode")
    executed_tail = (
        not episodes.unresolved.empty
        and episodes.unresolved["status"].astype(str).str.startswith("open_cash").any()
    )
    if executed_tail:
        raise BinaryLedgerError("Partially executed terminal CASH episode creates residual")
    ledger_edge = math.log(strategy_end.last_equity / strategy_start_state.last_equity) - math.log(
        benchmark_end.last_equity / buy_hold_start_state.last_equity
    )
    atomic_edge = math.fsum(
        float(value)
        for value in strategy["active_log_increment_vs_buy_hold"].tolist()
    ) - math.fsum(
        float(value)
        for value in buy_hold["active_log_increment_vs_buy_hold"].tolist()
    )
    if abs(ledger_edge - atomic_edge) > tolerance_value:
        raise BinaryLedgerError("Ledger active edge differs from exact atomic increments")
    episode_edge = math.fsum(
        float(value) for value in episodes.complete["net_active_log_edge"].tolist()
    )
    residual = ledger_edge - episode_edge
    if abs(residual) > tolerance_value:
        raise BinaryLedgerError("Complete CASH episodes leave a boundary residual")
    return {
        "passed": True,
        "ledger_active_log_edge": ledger_edge,
        "atomic_active_log_edge": atomic_edge,
        "complete_episode_active_log_edge": episode_edge,
        "reconciliation_error": residual,
        "tolerance": tolerance_value,
    }


XOR_COMPONENT_COLUMNS = (
    "fill_date",
    "reference_adjusted_open",
    "primary_before",
    "primary_after",
    "comparator_before",
    "comparator_after",
    "orientation_after",
    "raw_market_component",
    "transition_cost_component",
    "net_signed_log_edge",
)

XOR_COLUMNS = (
    "xor_id",
    "entry_fill_date",
    "exit_fill_date",
    "orientation",
    "xor_fill_observations",
    "raw_market_component",
    "transition_cost_component",
    "net_signed_log_edge",
    "xor_sha256",
)

UNRESOLVED_XOR_COLUMNS = (
    "status",
    "entry_fill_date",
    "last_fill_date",
    "orientation",
    "raw_market_component",
    "transition_cost_component",
    "net_signed_log_edge",
    "unresolved_sha256",
)


@dataclass(frozen=True)
class XorExtraction:
    components: pd.DataFrame
    complete: pd.DataFrame
    unresolved: pd.DataFrame


def extract_signed_xor_episodes(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    primary_start_state: AccountState,
    comparator_start_state: AccountState,
) -> XorExtraction:
    verify_ledger(primary, start_state=primary_start_state)
    verify_ledger(comparator, start_state=comparator_start_state)
    _aligned_ledgers(primary, comparator)
    cost = primary_start_state.cost_bps / 10_000.0
    components: list[dict[str, Any]] = []
    complete: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    run: dict[str, Any] | None = None
    previous_ref = primary_start_state.last_reference_price

    def orientation(primary_exposure: int, comparator_exposure: int) -> str | None:
        if primary_exposure == comparator_exposure:
            return None
        return (
            PRIMARY_CASH_COMPARATOR_LONG
            if primary_exposure == 0
            else PRIMARY_LONG_COMPARATOR_CASH
        )

    def start_run(
        *, fill_date: str, value: str, partial_left: bool
    ) -> dict[str, Any]:
        return {
            "entry_fill_date": fill_date,
            "last_fill_date": fill_date,
            "partial_left": partial_left,
            "orientation": value,
            "xor_fill_observations": 0,
            "raw": 0.0,
            "cost": 0.0,
            "net": 0.0,
        }

    def add_to_run(
        current: dict[str, Any],
        *,
        fill_date: str,
        raw_value: float,
        cost_value: float,
        after_is_different: bool,
    ) -> None:
        current["last_fill_date"] = fill_date
        current["raw"] += raw_value
        current["cost"] += cost_value
        current["net"] += raw_value + cost_value
        if after_is_different:
            current["xor_fill_observations"] += 1

    def finish_run(current: dict[str, Any], *, fill_date: str) -> None:
        value = current["orientation"]
        if value not in {
            PRIMARY_CASH_COMPARATOR_LONG,
            PRIMARY_LONG_COMPARATOR_CASH,
        }:
            raise BinaryLedgerError("XOR run has a mixed or invalid orientation")
        if current["partial_left"]:
            payload = {
                "status": "left_boundary_partial",
                "entry_fill_date": current["entry_fill_date"],
                "last_fill_date": fill_date,
                "orientation": value,
                "raw_market_component": current["raw"],
                "transition_cost_component": current["cost"],
                "net_signed_log_edge": current["net"],
            }
            unresolved.append({**payload, "unresolved_sha256": _sha256(payload)})
            return
        payload = {
            "entry_fill_date": current["entry_fill_date"],
            "exit_fill_date": fill_date,
            "orientation": value,
            "xor_fill_observations": current["xor_fill_observations"],
            "raw_market_component": current["raw"],
            "transition_cost_component": current["cost"],
            "net_signed_log_edge": current["net"],
        }
        xor_id = _sha256(
            {
                "primary": primary_start_state.policy_name,
                "comparator": comparator_start_state.policy_name,
                "entry": current["entry_fill_date"],
                "orientation": value,
            }
        )
        complete_payload = {"xor_id": xor_id, **payload}
        complete.append(
            {**complete_payload, "xor_sha256": _sha256(complete_payload)}
        )

    for primary_row, comparator_row in zip(
        primary.to_dict(orient="records"), comparator.to_dict(orient="records")
    ):
        ref = float(primary_row["reference_adjusted_open"])
        p_before = int(primary_row["held_exposure_before_fill"])
        c_before = int(comparator_row["held_exposure_before_fill"])
        p_after = int(primary_row["requested_target_exposure"])
        c_after = int(comparator_row["requested_target_exposure"])
        raw = (
            0.0
            if previous_ref is None
            else (p_before - c_before) * math.log(ref / previous_ref)
        )
        primary_transition = _transition_cost(p_before, p_after, cost)
        comparator_transition = -_transition_cost(c_before, c_after, cost)
        transition = primary_transition + comparator_transition
        net = raw + transition
        expected_net = float(
            primary_row["active_log_increment_vs_buy_hold"]
        ) - float(comparator_row["active_log_increment_vs_buy_hold"])
        if abs(net - expected_net) > 1e-15:
            raise BinaryLedgerError("Signed XOR component differs from ledger increments")
        before_orientation = orientation(p_before, c_before)
        after_orientation = orientation(p_after, c_after)
        orientation_after = "equal" if after_orientation is None else after_orientation
        component = {
            "fill_date": primary_row["fill_date"],
            "reference_adjusted_open": ref,
            "primary_before": p_before,
            "primary_after": p_after,
            "comparator_before": c_before,
            "comparator_after": c_after,
            "orientation_after": orientation_after,
            "raw_market_component": raw,
            "transition_cost_component": transition,
            "net_signed_log_edge": net,
        }
        components.append(component)

        fill_date = primary_row["fill_date"]
        if before_orientation is None and after_orientation is None:
            if run is not None:
                raise BinaryLedgerError("Equal XOR state retained an open run")
        elif before_orientation is None:
            if run is not None or after_orientation is None:
                raise BinaryLedgerError("XOR divergence state is inconsistent")
            run = start_run(
                fill_date=fill_date,
                value=after_orientation,
                partial_left=False,
            )
            add_to_run(
                run,
                fill_date=fill_date,
                raw_value=raw,
                cost_value=transition,
                after_is_different=True,
            )
        elif after_orientation is None:
            if run is None:
                run = start_run(
                    fill_date=fill_date,
                    value=before_orientation,
                    partial_left=True,
                )
            if run["orientation"] != before_orientation:
                raise BinaryLedgerError("XOR convergence changed orientation")
            add_to_run(
                run,
                fill_date=fill_date,
                raw_value=raw,
                cost_value=transition,
                after_is_different=False,
            )
            finish_run(run, fill_date=fill_date)
            run = None
        elif before_orientation == after_orientation:
            if run is None:
                run = start_run(
                    fill_date=fill_date,
                    value=before_orientation,
                    partial_left=True,
                )
            if run["orientation"] != before_orientation:
                raise BinaryLedgerError("XOR run changed orientation without a split")
            add_to_run(
                run,
                fill_date=fill_date,
                raw_value=raw,
                cost_value=transition,
                after_is_different=True,
            )
        else:
            # Both policies trade and the cash side swaps.  Close the old
            # directional run and open the new one at the same fill, assigning
            # each policy's changing-leg cost exactly once.
            if run is None:
                run = start_run(
                    fill_date=fill_date,
                    value=before_orientation,
                    partial_left=True,
                )
            if run["orientation"] != before_orientation:
                raise BinaryLedgerError("XOR flip begins from an invalid orientation")
            old_exit_cost = (
                primary_transition
                if before_orientation == PRIMARY_CASH_COMPARATOR_LONG
                else comparator_transition
            )
            new_entry_cost = (
                primary_transition
                if after_orientation == PRIMARY_CASH_COMPARATOR_LONG
                else comparator_transition
            )
            if abs((old_exit_cost + new_entry_cost) - transition) > 1e-15:
                raise BinaryLedgerError("XOR flip cost split is not exact")
            add_to_run(
                run,
                fill_date=fill_date,
                raw_value=raw,
                cost_value=old_exit_cost,
                after_is_different=False,
            )
            finish_run(run, fill_date=fill_date)
            run = start_run(
                fill_date=fill_date,
                value=after_orientation,
                partial_left=False,
            )
            add_to_run(
                run,
                fill_date=fill_date,
                raw_value=0.0,
                cost_value=new_entry_cost,
                after_is_different=True,
            )
        previous_ref = ref

    if run is not None:
        payload = {
            "status": (
                "both_boundaries_partial" if run["partial_left"] else "right_boundary_partial"
            ),
            "entry_fill_date": run["entry_fill_date"],
            "last_fill_date": run["last_fill_date"],
            "orientation": run["orientation"],
            "raw_market_component": run["raw"],
            "transition_cost_component": run["cost"],
            "net_signed_log_edge": run["net"],
        }
        unresolved.append({**payload, "unresolved_sha256": _sha256(payload)})
    return XorExtraction(
        components=pd.DataFrame(components, columns=XOR_COMPONENT_COLUMNS),
        complete=pd.DataFrame(complete, columns=XOR_COLUMNS),
        unresolved=pd.DataFrame(unresolved, columns=UNRESOLVED_XOR_COLUMNS),
    )


def reconcile_signed_xor(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    extraction: XorExtraction,
    *,
    primary_start_state: AccountState,
    comparator_start_state: AccountState,
    tolerance: float = RECONCILIATION_TOLERANCE,
) -> dict[str, Any]:
    tolerance_value = _finite_number(tolerance, field="tolerance", positive=True)
    primary_end = verify_ledger(primary, start_state=primary_start_state)
    comparator_end = verify_ledger(comparator, start_state=comparator_start_state)
    _aligned_ledgers(primary, comparator)
    if not isinstance(extraction, XorExtraction):
        raise BinaryLedgerError("Signed XOR evidence has the wrong type")
    regenerated = extract_signed_xor_episodes(
        primary,
        comparator,
        primary_start_state=primary_start_state,
        comparator_start_state=comparator_start_state,
    )
    _exact_frame_equal(
        extraction.components,
        regenerated.components,
        columns=XOR_COMPONENT_COLUMNS,
        field="Signed XOR component evidence",
    )
    _exact_frame_equal(
        extraction.complete,
        regenerated.complete,
        columns=XOR_COLUMNS,
        field="Complete signed XOR evidence",
    )
    _exact_frame_equal(
        extraction.unresolved,
        regenerated.unresolved,
        columns=UNRESOLVED_XOR_COLUMNS,
        field="Unresolved signed XOR evidence",
    )
    extraction = regenerated
    ledger_edge = math.log(primary_end.last_equity / primary_start_state.last_equity) - math.log(
        comparator_end.last_equity / comparator_start_state.last_equity
    )
    atomic_edge = math.fsum(
        float(value) for value in extraction.components["net_signed_log_edge"].tolist()
    )
    if abs(ledger_edge - atomic_edge) > tolerance_value:
        raise BinaryLedgerError("Signed XOR atomic increments do not match ledgers")
    complete_edge = math.fsum(
        float(value) for value in extraction.complete["net_signed_log_edge"].tolist()
    )
    residual = ledger_edge - complete_edge
    if not extraction.unresolved.empty:
        raise BinaryLedgerError("Partially executed XOR boundary creates residual")
    if abs(residual) > tolerance_value:
        raise BinaryLedgerError("Complete signed XOR runs leave a residual")
    return {
        "passed": True,
        "ledger_signed_log_edge": ledger_edge,
        "atomic_signed_log_edge": atomic_edge,
        "complete_xor_signed_log_edge": complete_edge,
        "reconciliation_error": residual,
        "tolerance": tolerance_value,
    }


__all__ = [
    "ACCOUNT_INCEPTION_DATE",
    "ACCOUNT_INCEPTION_FILL_DATE",
    "ACCOUNT_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "EPISODE_COLUMNS",
    "INITIAL_CASH",
    "LEDGER_COLUMNS",
    "PRIMARY_CASH_COMPARATOR_LONG",
    "PRIMARY_LONG_COMPARATOR_CASH",
    "RECONCILIATION_TOLERANCE",
    "UNRESOLVED_EPISODE_COLUMNS",
    "UNRESOLVED_XOR_COLUMNS",
    "XOR_COLUMNS",
    "XOR_COMPONENT_COLUMNS",
    "AccountState",
    "BinaryLedgerError",
    "EpisodeExtraction",
    "LedgerRun",
    "XorExtraction",
    "assert_always_long_matches_buy_hold",
    "assert_cross_cost_action_identity",
    "assert_no_leverage",
    "extract_cash_episodes",
    "extract_signed_xor_episodes",
    "reconcile_complete_cash_episodes",
    "reconcile_signed_xor",
    "run_continuous_ledger",
    "run_verified_continuation",
    "verify_continuation_checkpoint",
    "verify_ledger",
]
