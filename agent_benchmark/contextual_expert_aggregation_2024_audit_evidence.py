"""Ledger, cash-episode, and fill-based XOR evidence for the 2024 audit.

The functions here contain no performance gates and no expected result.  They
only continue already verified accounts and reconstruct deterministic economic
evidence from canonical ledgers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd

from . import contextual_expert_aggregation_ledger as _ledger


RECONCILIATION_TOLERANCE = 1e-10
SHADOW_CASH_LEAD_LONG = "shadow_cash_lead_long"
SHADOW_LONG_LEAD_CASH = "shadow_long_lead_cash"

AUDIT_XOR_COMPONENT_COLUMNS = (
    "fill_date",
    "reference_adjusted_open",
    "shadow_decision_date",
    "lead_decision_date",
    "shadow_before",
    "shadow_after",
    "lead_before",
    "lead_after",
    "orientation_after",
    "raw_market_component",
    "transition_cost_component",
    "net_incremental_log_edge",
    "component_sha256",
)
AUDIT_XOR_COLUMNS = (
    "xor_id",
    "entry_fill_date",
    "exit_fill_date",
    "entry_quarter",
    "start_shadow_decision_date",
    "start_lead_decision_date",
    "end_shadow_decision_date",
    "end_lead_decision_date",
    "orientation",
    "xor_fill_observations",
    "raw_market_component",
    "transition_cost_component",
    "net_incremental_log_edge",
    "xor_sha256",
)
AUDIT_OPEN_XOR_COLUMNS = (
    "status",
    "entry_fill_date",
    "last_fill_date",
    "entry_quarter",
    "start_shadow_decision_date",
    "start_lead_decision_date",
    "last_shadow_decision_date",
    "last_lead_decision_date",
    "orientation",
    "xor_fill_observations",
    "raw_market_component",
    "transition_cost_component",
    "net_incremental_log_edge",
    "open_xor_sha256",
)


class AuditEvidenceError(RuntimeError):
    """Raised when deterministic ledger evidence does not reconcile."""


@dataclass(frozen=True)
class ContinuedAccount:
    full_ledger: pd.DataFrame
    suffix_ledger: pd.DataFrame
    terminal_state: _ledger.AccountState
    no_leverage_proof: Mapping[str, Any]


@dataclass(frozen=True)
class CashEpisodeEvidence:
    complete: pd.DataFrame
    open: pd.DataFrame
    reconciliation: Mapping[str, Any]


@dataclass(frozen=True)
class AuditXorEvidence:
    components: pd.DataFrame
    complete: pd.DataFrame
    open: pd.DataFrame
    reconciliation: Mapping[str, Any]


def _canonical(value: Any) -> bytes:
    def clean(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): clean(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if hasattr(item, "item"):
            return clean(item.item())
        return item

    return json.dumps(
        clean(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical(value)).hexdigest()}"


def continue_verified_account(
    *,
    prior_ledger: pd.DataFrame,
    parent_state: _ledger.AccountState,
    suffix_adjusted_opens: pd.Series,
    suffix_close_targets: pd.Series,
) -> ContinuedAccount:
    """Verify the exact parent ledger, then execute a suffix without a reset."""

    initial = _ledger.AccountState.initial(
        policy_name=parent_state.policy_name,
        cost_bps=parent_state.cost_bps,
    )
    _ledger.verify_ledger(
        prior_ledger,
        start_state=initial,
        expected_end_state=parent_state,
    )
    inputs = _ledger._canonical_inputs(suffix_adjusted_opens, suffix_close_targets)
    run = _ledger._run_canonical_inputs(inputs, state=parent_state)
    full = pd.concat([prior_ledger, run.ledger], ignore_index=True)
    _ledger.verify_ledger(full, start_state=initial, expected_end_state=run.state)
    proof = _ledger.assert_no_leverage(full, start_state=initial)
    if (
        run.state.inception_count != 1
        or run.state.ledger_row_count != parent_state.ledger_row_count + len(run.ledger)
        or run.ledger.empty
        or run.ledger.iloc[0]["previous_row_sha256"]
        != parent_state.ledger_tip_sha256
    ):
        raise AuditEvidenceError("continued account reset or broke its hash chain")
    return ContinuedAccount(
        full_ledger=full,
        suffix_ledger=run.ledger,
        terminal_state=run.state,
        no_leverage_proof=proof,
    )


def cash_episode_evidence(
    *,
    ledger: pd.DataFrame,
    terminal_state: _ledger.AccountState,
) -> CashEpisodeEvidence:
    initial = _ledger.AccountState.initial(
        policy_name=terminal_state.policy_name,
        cost_bps=terminal_state.cost_bps,
    )
    extraction = _ledger.extract_cash_episodes(
        ledger,
        start_state=initial,
        expected_end_state=terminal_state,
    )
    ledger_edge = math.fsum(
        float(value) for value in ledger["active_log_increment_vs_buy_hold"].tolist()
    )
    complete_edge = math.fsum(
        float(value) for value in extraction.complete["net_active_log_edge"].tolist()
    )
    open_edge = math.fsum(
        float(value)
        for value in extraction.unresolved["net_active_log_edge_to_mark"].tolist()
    )
    error = ledger_edge - complete_edge - open_edge
    if abs(error) > RECONCILIATION_TOLERANCE:
        raise AuditEvidenceError("cash episodes do not reconcile the full ledger")
    return CashEpisodeEvidence(
        complete=extraction.complete,
        open=extraction.unresolved,
        reconciliation={
            "ledger_active_log_edge": ledger_edge,
            "complete_episode_active_log_edge": complete_edge,
            "open_episode_active_log_edge": open_edge,
            "reconciliation_error": error,
            "tolerance": RECONCILIATION_TOLERANCE,
            "terminal_episode_force_closed": False,
            "passed": True,
        },
    )


def _orientation(value: str) -> str:
    mapping = {
        _ledger.PRIMARY_CASH_COMPARATOR_LONG: SHADOW_CASH_LEAD_LONG,
        _ledger.PRIMARY_LONG_COMPARATOR_CASH: SHADOW_LONG_LEAD_CASH,
        "equal": "equal",
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise AuditEvidenceError("inherited XOR orientation changed") from exc


def _quarter(fill_date: str) -> str:
    stamp = pd.Timestamp(fill_date)
    return f"{stamp.year}Q{((stamp.month - 1) // 3) + 1}"


def extract_audit_xor_evidence(
    *,
    full_shadow: pd.DataFrame,
    full_lead: pd.DataFrame,
    shadow_parent_state: _ledger.AccountState,
    lead_parent_state: _ledger.AccountState,
    shadow_terminal_state: _ledger.AccountState,
    lead_terminal_state: _ledger.AccountState,
) -> AuditXorEvidence:
    """Build contract-specific fill XORs after verifying both full ledgers."""

    if shadow_parent_state.last_session_date != lead_parent_state.last_session_date:
        raise AuditEvidenceError("lead and shadow parent boundaries differ")
    for frame, parent, terminal in (
        (full_shadow, shadow_parent_state, shadow_terminal_state),
        (full_lead, lead_parent_state, lead_terminal_state),
    ):
        initial = _ledger.AccountState.initial(
            policy_name=parent.policy_name,
            cost_bps=parent.cost_bps,
        )
        _ledger.verify_ledger(frame, start_state=initial, expected_end_state=terminal)
    _ledger._aligned_ledgers(full_shadow, full_lead)

    boundary = str(shadow_parent_state.last_session_date)
    shadow = full_shadow.loc[full_shadow["fill_date"] > boundary].reset_index(drop=True)
    lead = full_lead.loc[full_lead["fill_date"] > boundary].reset_index(drop=True)
    if shadow.empty or len(shadow) != len(lead):
        raise AuditEvidenceError("lead and shadow suffix ledgers are not aligned")
    inherited = _ledger.extract_signed_xor_episodes(
        shadow,
        lead,
        primary_start_state=shadow_parent_state,
        comparator_start_state=lead_parent_state,
    )
    shadow_by_fill = shadow.set_index("fill_date", drop=False)
    lead_by_fill = lead.set_index("fill_date", drop=False)

    components: list[dict[str, Any]] = []
    for record in inherited.components.to_dict(orient="records"):
        fill = str(record["fill_date"])
        srow = shadow_by_fill.loc[fill]
        lrow = lead_by_fill.loc[fill]
        base = {
            "fill_date": fill,
            "reference_adjusted_open": float(record["reference_adjusted_open"]),
            "shadow_decision_date": str(srow["decision_date"]),
            "lead_decision_date": str(lrow["decision_date"]),
            "shadow_before": int(record["primary_before"]),
            "shadow_after": int(record["primary_after"]),
            "lead_before": int(record["comparator_before"]),
            "lead_after": int(record["comparator_after"]),
            "orientation_after": _orientation(str(record["orientation_after"])),
            "raw_market_component": float(record["raw_market_component"]),
            "transition_cost_component": float(record["transition_cost_component"]),
            "net_incremental_log_edge": float(record["net_signed_log_edge"]),
        }
        components.append({**base, "component_sha256": _sha256(base)})

    complete: list[dict[str, Any]] = []
    for record in inherited.complete.to_dict(orient="records"):
        entry = str(record["entry_fill_date"])
        exit_fill = str(record["exit_fill_date"])
        s_entry, l_entry = shadow_by_fill.loc[entry], lead_by_fill.loc[entry]
        s_exit, l_exit = shadow_by_fill.loc[exit_fill], lead_by_fill.loc[exit_fill]
        base = {
            "xor_id": str(record["xor_id"]),
            "entry_fill_date": entry,
            "exit_fill_date": exit_fill,
            "entry_quarter": _quarter(entry),
            "start_shadow_decision_date": str(s_entry["decision_date"]),
            "start_lead_decision_date": str(l_entry["decision_date"]),
            "end_shadow_decision_date": str(s_exit["decision_date"]),
            "end_lead_decision_date": str(l_exit["decision_date"]),
            "orientation": _orientation(str(record["orientation"])),
            "xor_fill_observations": int(record["xor_fill_observations"]),
            "raw_market_component": float(record["raw_market_component"]),
            "transition_cost_component": float(record["transition_cost_component"]),
            "net_incremental_log_edge": float(record["net_signed_log_edge"]),
        }
        complete.append({**base, "xor_sha256": _sha256(base)})

    opened: list[dict[str, Any]] = []
    for record in inherited.unresolved.to_dict(orient="records"):
        entry = str(record["entry_fill_date"])
        last = str(record["last_fill_date"])
        s_entry, l_entry = shadow_by_fill.loc[entry], lead_by_fill.loc[entry]
        s_last, l_last = shadow_by_fill.loc[last], lead_by_fill.loc[last]
        orientation = _orientation(str(record["orientation"]))
        observations = sum(
            1
            for item in components
            if entry <= item["fill_date"] <= last
            and item["orientation_after"] == orientation
        )
        base = {
            "status": str(record["status"]),
            "entry_fill_date": entry,
            "last_fill_date": last,
            "entry_quarter": _quarter(entry),
            "start_shadow_decision_date": str(s_entry["decision_date"]),
            "start_lead_decision_date": str(l_entry["decision_date"]),
            "last_shadow_decision_date": str(s_last["decision_date"]),
            "last_lead_decision_date": str(l_last["decision_date"]),
            "orientation": orientation,
            "xor_fill_observations": observations,
            "raw_market_component": float(record["raw_market_component"]),
            "transition_cost_component": float(record["transition_cost_component"]),
            "net_incremental_log_edge": float(record["net_signed_log_edge"]),
        }
        opened.append({**base, "open_xor_sha256": _sha256(base)})

    component_frame = pd.DataFrame(components, columns=AUDIT_XOR_COMPONENT_COLUMNS)
    complete_frame = pd.DataFrame(complete, columns=AUDIT_XOR_COLUMNS)
    open_frame = pd.DataFrame(opened, columns=AUDIT_OPEN_XOR_COLUMNS)
    full_edge = (
        shadow_terminal_state.cumulative_active_log_edge
        - shadow_parent_state.cumulative_active_log_edge
        - (
            lead_terminal_state.cumulative_active_log_edge
            - lead_parent_state.cumulative_active_log_edge
        )
    )
    component_edge = math.fsum(
        float(value) for value in component_frame["net_incremental_log_edge"].tolist()
    )
    complete_edge = math.fsum(
        float(value) for value in complete_frame["net_incremental_log_edge"].tolist()
    )
    open_edge = math.fsum(
        float(value) for value in open_frame["net_incremental_log_edge"].tolist()
    )
    component_error = full_edge - component_edge
    episode_error = full_edge - complete_edge - open_edge
    if (
        abs(component_error) > RECONCILIATION_TOLERANCE
        or abs(episode_error) > RECONCILIATION_TOLERANCE
    ):
        raise AuditEvidenceError("fill-based XOR evidence does not reconcile")
    economic_fill_count = sum(
        item["orientation_after"] != "equal" for item in components
    )
    return AuditXorEvidence(
        components=component_frame,
        complete=complete_frame,
        open=open_frame,
        reconciliation={
            "full_period_incremental_active_log_edge": full_edge,
            "component_incremental_active_log_edge": component_edge,
            "complete_xor_incremental_active_log_edge": complete_edge,
            "open_xor_incremental_active_log_edge": open_edge,
            "component_reconciliation_error": component_error,
            "episode_reconciliation_error": episode_error,
            "economic_xor_fill_count": economic_fill_count,
            "complete_xor_count": len(complete_frame),
            "open_xor_count": len(open_frame),
            "terminal_xor_force_closed": False,
            "tolerance": RECONCILIATION_TOLERANCE,
            "passed": True,
        },
    )


def terminal_account_economic_equality(
    always_long: _ledger.AccountState,
    buy_hold: _ledger.AccountState,
    *,
    tolerance: float = RECONCILIATION_TOLERANCE,
) -> dict[str, Any]:
    ignored = {"policy_name", "ledger_tip_sha256"}
    left, right = asdict(always_long), asdict(buy_hold)
    if set(left) != set(right):
        raise AuditEvidenceError("terminal account schemas differ")
    for field in sorted(set(left) - ignored):
        a, b = left[field], right[field]
        if isinstance(a, float) or isinstance(b, float):
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise AuditEvidenceError("terminal account field types differ")
            if abs(float(a) - float(b)) > tolerance:
                raise AuditEvidenceError("always-long terminal economics differ")
        elif type(a) is not type(b) or a != b:
            raise AuditEvidenceError("always-long terminal economics differ")
    return {
        "passed": True,
        "excluded_identity_fields": sorted(ignored),
        "tolerance": tolerance,
    }


__all__ = [
    "RECONCILIATION_TOLERANCE",
    "SHADOW_CASH_LEAD_LONG",
    "SHADOW_LONG_LEAD_CASH",
    "AUDIT_XOR_COMPONENT_COLUMNS",
    "AUDIT_XOR_COLUMNS",
    "AUDIT_OPEN_XOR_COLUMNS",
    "AuditEvidenceError",
    "ContinuedAccount",
    "CashEpisodeEvidence",
    "AuditXorEvidence",
    "continue_verified_account",
    "cash_episode_evidence",
    "extract_audit_xor_evidence",
    "terminal_account_economic_equality",
]
