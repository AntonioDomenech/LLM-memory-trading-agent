"""Pure chronological 2024 replay and same-ledger account continuation.

The caller supplies a verified through-2023 parent and a post-lock bounded
snapshot.  This module makes no filesystem, Git, network, sealing, or reporting
decisions and contains no expected 2024 result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import contextual_expert_aggregation_2024_audit_artifacts as _artifacts
from . import contextual_expert_aggregation_2024_audit_evidence as _evidence
from . import contextual_expert_aggregation_2024_audit_input as _input
from . import contextual_expert_aggregation_2024_audit_parent as _parent
from . import contextual_expert_aggregation_2024_audit_replay as _audit_replay
from . import contextual_expert_aggregation_ledger as _ledger
from . import contextual_expert_aggregation_replay as _replay
from . import contextual_expert_aggregation_stage as _stage


EXPECTED_FULL_LEDGER_ROWS = 5_033
EXPECTED_SUFFIX_ROWS = 252


class AuditComputationError(RuntimeError):
    """Raised when replay or account continuation violates the contract."""


@dataclass(frozen=True)
class AuditComputation:
    bounded: _input.BoundedAuditSnapshot
    parent: _parent.ParentBundleEvidence
    forks: _audit_replay.TwoScenarioFork
    fixed_features: pd.DataFrame
    forecasts: Mapping[str, pd.DataFrame]
    fixed_comparator_forecast: pd.DataFrame
    matured_lessons: Mapping[str, pd.DataFrame]
    state_weight_diagnostics: pd.DataFrame
    ledgers: Mapping[str, Mapping[str, pd.DataFrame]]
    suffix_ledgers: Mapping[str, Mapping[str, pd.DataFrame]]
    accounts: Mapping[str, Mapping[str, _ledger.AccountState]]
    episodes: Mapping[str, Mapping[str, _evidence.CashEpisodeEvidence]]
    xors: Mapping[str, _evidence.AuditXorEvidence]
    action_stream_sha256: Mapping[str, str]
    replay_diagnostics: Mapping[str, Any]
    integrity_proofs: Mapping[str, bool]


def _jsonable(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            raise AuditComputationError("computation contains a nonfinite value")
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        if not math.isfinite(result):
            raise AuditComputationError("computation contains a nonfinite value")
        return result
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    raise AuditComputationError("computation contains an unsupported value type")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _policy_target(
    policy: str,
    *,
    forecasts: Mapping[str, pd.DataFrame],
    fixed: pd.DataFrame,
) -> pd.Series:
    if policy in _audit_replay.SCENARIO_ORDER:
        return forecasts[policy]["learner_target_exposure"].copy()
    if policy == "aapl_buy_hold":
        return pd.Series(1, index=fixed.index, dtype=int)
    column = {
        "always_long": "fixed_always_long_target_exposure",
        "exact_union_cash": "fixed_union_cash_target_exposure",
        "contextual_only": "fixed_contextual_only_target_exposure",
    }.get(policy)
    if column is None:
        raise AuditComputationError("unknown 2024 audit policy")
    return fixed[column].copy()


def _diagnostic_frame(
    forecasts: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    selected = [
        name for name in _stage.DIAGNOSTIC_COLUMNS if name not in {"arm", "decision_date"}
    ]
    frames: list[pd.DataFrame] = []
    for scenario in _audit_replay.SCENARIO_ORDER:
        value = forecasts[scenario].loc[:, selected].copy()
        value.insert(
            0,
            "decision_date",
            value.index.map(lambda item: item.date().isoformat()),
        )
        value.insert(0, "arm", scenario)
        frames.append(value.reset_index(drop=True).loc[:, list(_stage.DIAGNOSTIC_COLUMNS)])
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 2 * EXPECTED_SUFFIX_ROWS:
        raise AuditComputationError("state/weight diagnostic row count changed")
    return result


def _action_hash(frame: pd.DataFrame, *, column: str) -> str:
    return _sha256_json(
        [
            {
                "date": index.date().isoformat(),
                "target_exposure": int(value),
            }
            for index, value in frame[column].items()
        ]
    )


def _run_accounts(
    *,
    parent: _parent.ParentBundleEvidence,
    suffix: pd.DataFrame,
    forecasts: Mapping[str, pd.DataFrame],
    fixed: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, _ledger.AccountState]],
    dict[str, dict[str, _evidence.CashEpisodeEvidence]],
    dict[str, _evidence.AuditXorEvidence],
    dict[str, bool],
]:
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    suffix_ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    accounts: dict[str, dict[str, _ledger.AccountState]] = {}
    episodes: dict[str, dict[str, _evidence.CashEpisodeEvidence]] = {}
    proofs: dict[str, bool] = {}
    for cost in _artifacts.COST_ORDER:
        ledgers[cost] = {}
        suffix_ledgers[cost] = {}
        accounts[cost] = {}
        episodes[cost] = {}
        for policy in _artifacts.POLICY_ORDER:
            parent_state = parent.account_for(cost, policy)
            continued = _evidence.continue_verified_account(
                prior_ledger=parent.ledger_for(cost, policy),
                parent_state=parent_state,
                suffix_adjusted_opens=suffix["aapl_adj_open"],
                suffix_close_targets=_policy_target(
                    policy, forecasts=forecasts, fixed=fixed
                ),
            )
            if (
                len(continued.full_ledger) != EXPECTED_FULL_LEDGER_ROWS
                or len(continued.suffix_ledger) != EXPECTED_SUFFIX_ROWS
                or continued.terminal_state.ledger_row_count
                != EXPECTED_FULL_LEDGER_ROWS
            ):
                raise AuditComputationError("continuous account row boundary changed")
            ledgers[cost][policy] = continued.full_ledger
            suffix_ledgers[cost][policy] = continued.suffix_ledger
            accounts[cost][policy] = continued.terminal_state
            episodes[cost][policy] = _evidence.cash_episode_evidence(
                ledger=continued.full_ledger,
                terminal_state=continued.terminal_state,
            )
            proofs[f"{cost}__{policy}__no_leverage"] = bool(
                continued.no_leverage_proof["passed"]
            )
            proofs[f"{cost}__{policy}__cash_episode_reconciliation"] = bool(
                episodes[cost][policy].reconciliation["passed"]
            )

        try:
            _ledger.assert_always_long_matches_buy_hold(
                ledgers[cost]["always_long"], ledgers[cost]["aapl_buy_hold"]
            )
            _evidence.terminal_account_economic_equality(
                accounts[cost]["always_long"], accounts[cost]["aapl_buy_hold"]
            )
        except (_ledger.BinaryLedgerError, _evidence.AuditEvidenceError) as exc:
            raise AuditComputationError(
                "always-long economics differ from same-ledger AAPL"
            ) from exc
        proofs[f"{cost}__always_long_aapl_economic_equality"] = True

    for policy in _artifacts.POLICY_ORDER:
        try:
            _ledger.assert_cross_cost_action_identity(
                {cost: ledgers[cost][policy] for cost in _artifacts.COST_ORDER}
            )
        except _ledger.BinaryLedgerError as exc:
            raise AuditComputationError("policy action stream changed across costs") from exc
        proofs[f"{policy}__cross_cost_action_identity"] = True

    xors: dict[str, _evidence.AuditXorEvidence] = {}
    for cost in _artifacts.COST_ORDER:
        xors[cost] = _evidence.extract_audit_xor_evidence(
            full_shadow=ledgers[cost][_audit_replay.ONLINE_2024_SHADOW],
            full_lead=ledgers[cost][_audit_replay.FROZEN_2023_LEAD],
            shadow_parent_state=parent.account_for(
                cost, _audit_replay.ONLINE_2024_SHADOW
            ),
            lead_parent_state=parent.account_for(cost, _audit_replay.FROZEN_2023_LEAD),
            shadow_terminal_state=accounts[cost][
                _audit_replay.ONLINE_2024_SHADOW
            ],
            lead_terminal_state=accounts[cost][_audit_replay.FROZEN_2023_LEAD],
        )
        proofs[f"{cost}__fill_based_xor_reconciliation"] = bool(
            xors[cost].reconciliation["passed"]
        )
    return ledgers, suffix_ledgers, accounts, episodes, xors, proofs


def compute_2024_continuation(
    *,
    parent: _parent.ParentBundleEvidence,
    bounded: _input.BoundedAuditSnapshot,
) -> AuditComputation:
    if not isinstance(parent, _parent.ParentBundleEvidence) or not parent.verified:
        raise AuditComputationError("2024 computation requires the exact verified parent")
    if not isinstance(bounded, _input.BoundedAuditSnapshot):
        raise AuditComputationError("2024 computation requires the post-lock bounded input")
    forks = _audit_replay.fork_2024_scenarios(
        bounded.suffix,
        parent.checkpoint,
        historical_prefix=bounded.historical_prefix,
    )
    scenarios = dict(forks.scenarios)
    lead = scenarios[_audit_replay.FROZEN_2023_LEAD]
    shadow = scenarios[_audit_replay.ONLINE_2024_SHADOW]
    if not lead.opportunity_frame.equals(shadow.opportunity_frame):
        raise AuditComputationError("scenario fixed features differ")
    fixed = lead.opportunity_frame.copy()
    forecasts = {name: scenarios[name].forecast.copy() for name in _audit_replay.SCENARIO_ORDER}
    matured = {
        name: scenarios[name].matured_lessons.copy()
        for name in _audit_replay.SCENARIO_ORDER
    }
    if (
        len(fixed) != EXPECTED_SUFFIX_ROWS
        or not fixed.index.equals(bounded.suffix.index)
        or any(not forecasts[name].index.equals(fixed.index) for name in forecasts)
    ):
        raise AuditComputationError("2024 replay table boundary changed")
    comparator = fixed.loc[:, list(_replay.COMPARATOR_TARGET_COLUMNS[:3])].copy()
    expected_comparator_columns = (
        "fixed_always_long_target_exposure",
        "fixed_union_cash_target_exposure",
        "fixed_contextual_only_target_exposure",
    )
    comparator = comparator.loc[:, list(expected_comparator_columns)]
    action_hashes = {
        name: _action_hash(forecasts[name], column="learner_target_exposure")
        for name in _audit_replay.SCENARIO_ORDER
    }
    action_hashes["fixed_comparators"] = _sha256_json(
        [
            {
                "date": index.date().isoformat(),
                **{column: int(row[column]) for column in comparator.columns},
            }
            for index, row in comparator.iterrows()
        ]
    )

    ledgers, suffix_ledgers, accounts, episodes, xors, account_proofs = _run_accounts(
        parent=parent,
        suffix=bounded.suffix,
        forecasts=forecasts,
        fixed=fixed,
    )
    parent_admitted = parent.checkpoint.model_payload["state"][
        "admitted_lesson_count"
    ]
    replay_diagnostics = {
        "replay_diagnostics_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "stage": _artifacts.AUDIT_STAGE,
        "scenario_order": list(_audit_replay.SCENARIO_ORDER),
        "parent_checkpoint_sha256": forks.parent_checkpoint_sha256,
        "dedicated_two_scenario_fork": True,
        "inherited_four_arm_helper_used": False,
        "common_fixed_features": True,
        "by_scenario": {
            name: {
                "scenario_name": name,
                "initial_checkpoint_sha256": parent.checkpoint.digest_sha256,
                "initial_model_state_sha256": parent.checkpoint.model_state_sha256,
                "initial_pending_state_sha256": parent.checkpoint.pending_state_sha256,
                "runtime": dict(scenarios[name].checkpoint.model_payload["runtime"]),
                "diagnostics": _jsonable(asdict(scenarios[name].diagnostics)),
                "terminal_checkpoint_sha256": scenarios[name].checkpoint.digest_sha256,
                "terminal_model_state_sha256": scenarios[name].checkpoint.model_state_sha256,
                "terminal_pending_state_sha256": scenarios[name].checkpoint.pending_state_sha256,
                "action_stream_sha256": action_hashes[name],
            }
            for name in _audit_replay.SCENARIO_ORDER
        },
        "frozen_post_cutoff_admitted_events": lead.diagnostics.admitted_events,
        "frozen_terminal_admitted_count_equals_parent": (
            lead.checkpoint.model_payload["state"]["admitted_lesson_count"]
            == parent_admitted
        ),
        "shadow_causal_maturity_order_inherited": True,
    }
    integrity = {
        "exact_parent_model_and_account_fork": True,
        "no_account_reset_or_added_capital": True,
        "frozen_lead_zero_post_cutoff_admissions": lead.diagnostics.admitted_events == 0,
        "online_shadow_causal_runtime": shadow.checkpoint.model_payload["runtime"]
        == {
            "learning_mode": _replay.CAUSAL_ONLINE_MODE,
            "frozen_cutoff": None,
            "ablation_mode": _replay.FULL_MODE,
        },
        "binary_forecast_targets": all(
            set(forecasts[name]["learner_target_exposure"].astype(int).tolist())
            <= {0, 1}
            for name in forecasts
        ),
        "fixed_features_equal_between_scenarios": True,
        "forecast_frozen_before_ledger_interpretation": True,
        **account_proofs,
    }
    return AuditComputation(
        bounded=bounded,
        parent=parent,
        forks=forks,
        fixed_features=fixed,
        forecasts=forecasts,
        fixed_comparator_forecast=comparator,
        matured_lessons=matured,
        state_weight_diagnostics=_diagnostic_frame(forecasts),
        ledgers=ledgers,
        suffix_ledgers=suffix_ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        action_stream_sha256=action_hashes,
        replay_diagnostics=replay_diagnostics,
        integrity_proofs=integrity,
    )


__all__ = [
    "EXPECTED_FULL_LEDGER_ROWS",
    "EXPECTED_SUFFIX_ROWS",
    "AuditComputationError",
    "AuditComputation",
    "compute_2024_continuation",
]
