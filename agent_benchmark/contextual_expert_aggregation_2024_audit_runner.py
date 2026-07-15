"""One-shot stage runner for the preregistered frozen-policy 2024 audit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import contextual_expert_aggregation_2024_audit_artifacts as _artifacts
from . import contextual_expert_aggregation_2024_audit_computation as _computation
from . import contextual_expert_aggregation_2024_audit_control as _control
from . import contextual_expert_aggregation_2024_audit_evaluation as _evaluation
from . import contextual_expert_aggregation_2024_audit_evidence as _evidence
from . import contextual_expert_aggregation_2024_audit_input as _input
from . import contextual_expert_aggregation_2024_audit_parent as _parent
from . import contextual_expert_aggregation_2024_audit_replay as _audit_replay
from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_ledger as _ledger


EVIDENCE_CLASSIFICATION = "post_hoc_frozen_policy_2024_replication"
RUNTIME_EVIDENCE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
PREPROMOTION_DEADLINE_SECONDS = (
    _artifacts.STAGE_DEADLINE_SECONDS - _artifacts.FINALIZATION_RESERVE_SECONDS
)


class ContextualExpertAggregation2024AuditError(RuntimeError):
    """The one-shot 2024 stage failed closed."""


@dataclass
class RuntimeClock:
    start: float
    clock: Callable[[], float] = time.monotonic
    samples: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if type(self.start) is not float or not math.isfinite(self.start):
            raise ContextualExpertAggregation2024AuditError(
                "bootstrap monotonic start is invalid"
            )
        if self.samples is None:
            self.samples = {}

    def elapsed(self) -> float:
        value = float(self.clock()) - self.start
        if not math.isfinite(value) or value < 0.0:
            raise ContextualExpertAggregation2024AuditError(
                "monotonic audit clock moved backwards or became nonfinite"
            )
        return value

    def sample(self, name: str, *, limit: float) -> float:
        assert self.samples is not None
        if name in self.samples:
            raise ContextualExpertAggregation2024AuditError(
                "runtime phase was sampled more than once"
            )
        elapsed = self.elapsed()
        if not elapsed < limit:
            raise ContextualExpertAggregation2024AuditError(
                "strict stage runtime deadline was exceeded"
            )
        if self.samples and elapsed <= self.samples[next(reversed(self.samples))]:
            raise ContextualExpertAggregation2024AuditError(
                "runtime samples are not strictly increasing"
            )
        self.samples[name] = elapsed
        return elapsed

    def require(self, *, limit: float) -> float:
        elapsed = self.elapsed()
        if not elapsed < limit:
            raise ContextualExpertAggregation2024AuditError(
                "strict stage runtime deadline was exceeded"
            )
        if self.samples and elapsed <= self.samples[next(reversed(self.samples))]:
            raise ContextualExpertAggregation2024AuditError(
                "live runtime guard is not after the prior named sample"
            )
        return elapsed


@dataclass(frozen=True)
class StageContext:
    repo_root: Path
    bootstrap_attestation: Mapping[str, Any]
    git_identity: Mapping[str, Any]
    lock: _control.AttemptLockEvidence
    parent: _parent.ParentBundleEvidence
    bounded: _input.BoundedAuditSnapshot
    postlock_tracking: Mapping[str, Any]
    computation: _computation.AuditComputation
    evaluation: Mapping[str, Any]
    integrity_checks: Mapping[str, bool]


def _jsonable(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            raise ContextualExpertAggregation2024AuditError(
                "audit payload contains a nonfinite value"
            )
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ContextualExpertAggregation2024AuditError(
                "audit payload contains a nonfinite value"
            )
        return numeric
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    raise ContextualExpertAggregation2024AuditError(
        "audit payload contains an unsupported value type"
    )


def _canonical_json_bytes_without_lf(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes_without_lf(value))


def _self_hashed(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if field in value:
        raise ContextualExpertAggregation2024AuditError(
            "self-hash field must be omitted from its unsigned domain"
        )
    clean = _jsonable(dict(value))
    return {**clean, field: _sha256_json(clean)}


def _json_payload(value: Any) -> bytes:
    return _artifacts.canonical_json_line_bytes(_jsonable(value))


def _build_checkpoint(computation: _computation.AuditComputation) -> dict[str, Any]:
    parent = computation.parent
    unsigned = {
        "checkpoint_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "checkpoint_cutoff": "2024-12-31",
        "source_start_session": "1999-03-10",
        "source_session_count": 6496,
        "scenario_order": list(_artifacts.SCENARIO_ORDER),
        "cost_order": list(_artifacts.COST_ORDER),
        "policy_order": list(_artifacts.POLICY_ORDER),
        "parent_identity": {
            "manifest_file_sha256": parent.manifest_file_sha256,
            "manifest_self_sha256": parent.manifest_self_sha256,
            "checkpoint_file_sha256": parent.checkpoint_file_sha256,
            "checkpoint_self_sha256": parent.checkpoint_self_sha256,
            "selected_checkpoint_sha256": parent.checkpoint.digest_sha256,
            "through_2023_canonical_sha256": _input.PREFIX_CANONICAL_SHA256,
        },
        "replay_checkpoints": {
            scenario: computation.forks.scenarios[scenario].checkpoint.to_dict()
            for scenario in _artifacts.SCENARIO_ORDER
        },
        "administrative_accounts": {
            cost: {
                policy: computation.accounts[cost][policy].to_checkpoint()
                for policy in _artifacts.POLICY_ORDER
            }
            for cost in _artifacts.COST_ORDER
        },
        "parent_administrative_account_identity": {
            cost: {
                policy: {
                    "account_state_sha256": parent.account_checkpoint_sha256[cost][
                        parent._policy(policy)
                    ],
                    "ledger_tip_sha256": parent.account_for(
                        cost, policy
                    ).ledger_tip_sha256,
                    "ledger_row_count": parent.account_for(cost, policy).ledger_row_count,
                }
                for policy in _artifacts.POLICY_ORDER
            }
            for cost in _artifacts.COST_ORDER
        },
    }
    return _self_hashed(unsigned, field="checkpoint_sha256")


def _pending_payload(computation: _computation.AuditComputation) -> dict[str, Any]:
    unsigned = {
        "pending_artifact_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "stage": _artifacts.AUDIT_STAGE,
        "scenario_order": list(_artifacts.SCENARIO_ORDER),
        "by_scenario": {
            scenario: {
                "pending_lessons": _jsonable(
                    list(computation.forks.scenarios[scenario].checkpoint.pending_lessons)
                ),
                "pending_lesson_count": len(
                    computation.forks.scenarios[scenario].checkpoint.pending_lessons
                ),
                "pending_state_sha256": computation.forks.scenarios[
                    scenario
                ].checkpoint.pending_state_sha256,
                "signal_cooldown_tail": _jsonable(
                    list(
                        computation.forks.scenarios[
                            scenario
                        ].checkpoint.signal_cooldown_tail
                    )
                ),
                "signal_cooldown_tail_sha256": computation.forks.scenarios[
                    scenario
                ].checkpoint.signal_cooldown_tail_sha256,
                "cooldown_predecessor": _jsonable(
                    dict(
                        computation.forks.scenarios[
                            scenario
                        ].checkpoint.cooldown_predecessor
                    )
                ),
            }
            for scenario in _artifacts.SCENARIO_ORDER
        },
    }
    return _self_hashed(unsigned, field="artifact_sha256")


def _episodes_payload(
    computation: _computation.AuditComputation, *, cost: str
) -> dict[str, Any]:
    unsigned = {
        "episode_artifact_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "stage": _artifacts.AUDIT_STAGE,
        "cost": cost,
        "cost_bps": _evaluation.COST_BPS[cost],
        "policy_order": list(_artifacts.POLICY_ORDER),
        "policies": {
            policy: {
                "complete_columns": list(_ledger.EPISODE_COLUMNS),
                "complete": _jsonable(
                    computation.episodes[cost][policy].complete.to_dict(
                        orient="records"
                    )
                ),
                "open_columns": list(_ledger.UNRESOLVED_EPISODE_COLUMNS),
                "open": _jsonable(
                    computation.episodes[cost][policy].open.to_dict(orient="records")
                ),
                "reconciliation": _jsonable(
                    dict(computation.episodes[cost][policy].reconciliation)
                ),
            }
            for policy in _artifacts.POLICY_ORDER
        },
    }
    return _self_hashed(unsigned, field="artifact_sha256")


def _xor_payload(
    computation: _computation.AuditComputation, *, cost: str
) -> dict[str, Any]:
    extraction = computation.xors[cost]
    unsigned = {
        "xor_artifact_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "stage": _artifacts.AUDIT_STAGE,
        "cost": cost,
        "cost_bps": _evaluation.COST_BPS[cost],
        "orientation_order": [
            _evidence.SHADOW_CASH_LEAD_LONG,
            _evidence.SHADOW_LONG_LEAD_CASH,
        ],
        "component_columns": list(_evidence.AUDIT_XOR_COMPONENT_COLUMNS),
        "components": _jsonable(extraction.components.to_dict(orient="records")),
        "complete_columns": list(_evidence.AUDIT_XOR_COLUMNS),
        "complete": _jsonable(extraction.complete.to_dict(orient="records")),
        "open_columns": list(_evidence.AUDIT_OPEN_XOR_COLUMNS),
        "open": _jsonable(extraction.open.to_dict(orient="records")),
        "reconciliation": _jsonable(dict(extraction.reconciliation)),
    }
    return _self_hashed(unsigned, field="artifact_sha256")


def _prefix_payload(bounded: _input.BoundedAuditSnapshot) -> dict[str, Any]:
    unsigned = {
        **dict(bounded.prefix_continuity),
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "evidence_classification": "canonical_prefix_and_checkpoint_continuity",
    }
    return _self_hashed(unsigned, field="proof_sha256")


def _source_provenance_payload(
    *,
    parent: _parent.ParentBundleEvidence,
    bounded: _input.BoundedAuditSnapshot,
    lock: _control.AttemptLockEvidence,
) -> dict[str, Any]:
    unsigned = {
        "provenance_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "evidence_classification": "authorized_source_lineage",
        "parent": {
            "directory": parent.parent_directory,
            "manifest_file_sha256": parent.manifest_file_sha256,
            "manifest_self_sha256": parent.manifest_self_sha256,
            "checksums_file_sha256": parent.checksums_file_sha256,
            "checkpoint_file_sha256": parent.checkpoint_file_sha256,
            "checkpoint_self_sha256": parent.checkpoint_self_sha256,
            "selected_checkpoint_sha256": parent.checkpoint.digest_sha256,
        },
        "bounded_input": {
            "path": _input.INPUT_PATH.as_posix(),
            "raw_sha256": bounded.snapshot.raw_sha256,
            "canonical_sha256": bounded.snapshot.spec.canonical_sha256,
            "date_sequence_sha256": bounded.snapshot.spec.date_sequence_sha256,
            "first_session": bounded.snapshot.spec.first_session,
            "last_session": bounded.snapshot.spec.last_session,
            "rows": bounded.snapshot.spec.rows,
            "physical_snapshot_has_later_rows": False,
            "rows_after_bound_returned": False,
        },
        "sanitized_receipt": {
            "path": _input.RECEIPT_PATH.as_posix(),
            "raw_sha256": _sha256_bytes(bounded.receipt_bytes),
            "classification": bounded.receipt["classification"],
            "sealed_preparation_lineage": dict(
                bounded.receipt["sealed_preparation_lineage"]
            ),
        },
        "attempt_lock": {
            "path": _artifacts.ATTEMPT_LOCK_PATH.as_posix(),
            "raw_sha256": lock.raw_sha256,
            "created_before_market_release": True,
        },
        "access_boundary": {
            "receipt_read_after_lock": True,
            "input_read_after_receipt_and_lock": True,
            "suffix_released_after_prefix_proof": True,
            "later_market_data_accessed": False,
            "prior_2024_artifact_accessed": False,
        },
    }
    return _self_hashed(unsigned, field="provenance_sha256")


def _known_result_isolation_payload(
    *, git_identity: Mapping[str, Any]
) -> dict[str, Any]:
    paths = tuple(
        sorted(git_identity["dependency_identity"]["files"])
    )
    forbidden_fragments = (
        "2024_stage_success",
        "2024_result",
        "2024_action",
        "2024_forecast",
        "2024_ledger",
    )
    clean_paths = all(
        not any(fragment in path.lower() for fragment in forbidden_fragments)
        for path in paths
    )
    if not clean_paths:
        raise ContextualExpertAggregation2024AuditError(
            "dependency inventory contains a prior-2024 result surface"
        )
    unsigned = {
        "isolation_evidence_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "evidence_classification": "known_result_access_firewall",
        "dependency_paths": list(paths),
        "dependency_paths_sha256": _sha256_json(list(paths)),
        "forbidden_path_fragments": list(forbidden_fragments),
        "dependency_inventory_excludes_prior_2024_results": clean_paths,
        "prior_2024_forecasts_accessed": False,
        "prior_2024_actions_accessed": False,
        "prior_2024_ledgers_accessed": False,
        "prior_2024_metrics_accessed": False,
        "prior_2024_gates_accessed": False,
        "later_market_data_accessed": False,
        "expected_2024_result_fixture_used": False,
        "reporting_module_imported_by_policy_replay": False,
    }
    return _self_hashed(unsigned, field="isolation_evidence_sha256")


def _replay_payload(computation: _computation.AuditComputation) -> dict[str, Any]:
    return _self_hashed(
        dict(computation.replay_diagnostics), field="replay_diagnostics_sha256"
    )


def _metrics_payload(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "metrics_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "evidence_classification": evaluation["evidence_classification"],
        "metrics": _jsonable(evaluation["metrics"]),
    }
    return _self_hashed(unsigned, field="metrics_sha256")


def _gate_payload(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "gate_report_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "evidence_classification": evaluation["evidence_classification"],
        "gate_report": _jsonable(evaluation["gate_report"]),
    }
    return _self_hashed(unsigned, field="gate_report_sha256")


_STATIC_INTEGRITY_CHECKS = (
    "binary_actions",
    "bounded_input_identity",
    "contract_branch_pushed_dependency_identity",
    "control_root_and_output_identity",
    "durable_attempt_lock_identity",
    "exact_parent_bundle_checkpoint_accounts",
    "exact_parent_model_and_account_fork",
    "exact_prefix_checkpoint_continuity",
    "frozen_lead_zero_post_cutoff_admissions",
    "ledger_episode_xor_gate_reconciliation",
    "no_account_reset_or_added_capital",
    "no_later_market_data_access",
    "no_prior_2024_artifact_access",
    "no_short_leverage_borrowing_negative_cash",
    "one_lesson_per_eligible_opportunity",
    "online_shadow_causal_maturity_and_update_order",
    "pending_and_cooldown_state_exact",
    "sanitized_receipt_identity",
    "stage_runtime_protocol_bound",
    "zero_external_use",
)

_EXPECTED_COMPUTATION_PROOFS = frozenset(
    {
        "exact_parent_model_and_account_fork",
        "no_account_reset_or_added_capital",
        "frozen_lead_zero_post_cutoff_admissions",
        "online_shadow_causal_runtime",
        "binary_forecast_targets",
        "fixed_features_equal_between_scenarios",
        "forecast_frozen_before_ledger_interpretation",
        *(
            f"{cost}__{policy}__no_leverage"
            for cost in _artifacts.COST_ORDER
            for policy in _artifacts.POLICY_ORDER
        ),
        *(
            f"{cost}__{policy}__cash_episode_reconciliation"
            for cost in _artifacts.COST_ORDER
            for policy in _artifacts.POLICY_ORDER
        ),
        *(
            f"{cost}__always_long_aapl_economic_equality"
            for cost in _artifacts.COST_ORDER
        ),
        *(
            f"{policy}__cross_cost_action_identity"
            for policy in _artifacts.POLICY_ORDER
        ),
        *(
            f"{cost}__fill_based_xor_reconciliation"
            for cost in _artifacts.COST_ORDER
        ),
    }
)


def _checkpoint_valid(checkpoint: Any) -> bool:
    try:
        checkpoint.validate()
    except (TypeError, ValueError):
        return False
    return True


def _lesson_conservation(
    scenario: Any, *, parent_checkpoint: Any
) -> bool:
    parent_pending = [
        str(value["signal_date"]) for value in parent_checkpoint.pending_lessons
    ]
    opportunity_dates = [
        index.date().isoformat()
        for index, selected in scenario.opportunity_frame[
            "unfiltered_union_signal"
        ].items()
        if bool(selected)
    ]
    matured_dates = [
        str(value) for value in scenario.matured_lessons["signal_date"].tolist()
    ]
    terminal_pending = [
        str(value["signal_date"])
        for value in scenario.checkpoint.pending_lessons
    ]
    expected = parent_pending + opportunity_dates
    observed = matured_dates + terminal_pending
    parent_state = parent_checkpoint.model_payload["state"]
    terminal_state = scenario.checkpoint.model_payload["state"]
    admitted = (
        int(scenario.matured_lessons["admitted"].sum())
        if len(scenario.matured_lessons)
        else 0
    )
    return bool(
        len(expected) == len(set(expected))
        and len(matured_dates) == len(set(matured_dates))
        and len(terminal_pending) == len(set(terminal_pending))
        and set(matured_dates).isdisjoint(terminal_pending)
        and set(expected) == set(observed)
        and len(parent_pending) + scenario.diagnostics.accepted_opportunities
        == scenario.diagnostics.matured_events
        + scenario.diagnostics.ending_pending_lessons
        and scenario.diagnostics.accepted_opportunities == len(opportunity_dates)
        and scenario.diagnostics.matured_events == len(matured_dates)
        and scenario.diagnostics.ending_pending_lessons == len(terminal_pending)
        and scenario.diagnostics.admitted_events == admitted
        and terminal_state["processed_session_count"]
        == parent_state["processed_session_count"]
        + scenario.diagnostics.processed_rows
        and terminal_state["matured_lesson_count"]
        == parent_state["matured_lesson_count"]
        + scenario.diagnostics.matured_events
        and terminal_state["admitted_lesson_count"]
        == parent_state["admitted_lesson_count"]
        + scenario.diagnostics.admitted_events
    )


def _causal_update_order(
    scenario: Any,
    *,
    parent_checkpoint: Any,
    historical_prefix: pd.DataFrame,
) -> bool:
    forecast = scenario.forecast
    matured = scenario.matured_lessons
    if not forecast.index.is_monotonic_increasing or forecast.index.has_duplicates:
        return False
    combined_index = historical_prefix.index.append(forecast.index)
    if combined_index.has_duplicates or not combined_index.is_monotonic_increasing:
        return False
    positions = {
        value.date().isoformat(): index
        for index, value in enumerate(combined_index)
    }
    close_counts = forecast["matured_on_close_count"].astype(int)
    if not set(close_counts.tolist()) <= {0, 1} or int(close_counts.sum()) != len(
        matured
    ):
        return False
    parent_state = parent_checkpoint.model_payload["state"]
    expected_matured = (
        int(parent_state["matured_lesson_count"]) + close_counts.cumsum()
    )
    expected_admitted = int(parent_state["admitted_lesson_count"]) + forecast[
        "matured_admitted"
    ].astype(int).cumsum()
    expected_pending = (
        len(parent_checkpoint.pending_lessons)
        + scenario.opportunity_frame["unfiltered_union_signal"].astype(int).cumsum()
        - close_counts.cumsum()
    )
    if not (
        forecast["matured_lesson_count"].astype(int).equals(expected_matured)
        and forecast["admitted_lesson_count"].astype(int).equals(expected_admitted)
        and forecast["pending_lesson_count"].astype(int).equals(expected_pending)
    ):
        return False
    row_mapping = {
        "signal_date": "matured_signal_date",
        "signal_market_state": "matured_signal_market_state",
        "entry_adjusted_open": "matured_entry_adjusted_open",
        "exit_adjusted_open": "matured_exit_adjusted_open",
        "net_cash_log_edge_10bps": "matured_net_cash_log_edge_10bps",
        "reward_range": "matured_reward_range",
        "admitted": "matured_admitted",
        **{
            f"advice__{name}": f"matured_advice__{name}"
            for name in ("always_long", "union_cash", "contextual_only", "weak_trend_only")
        },
        **{
            f"reward__{name}": f"matured_reward__{name}"
            for name in ("always_long", "union_cash", "contextual_only", "weak_trend_only")
        },
    }
    for event in matured.to_dict(orient="records"):
        maturity_date = str(event["maturity_date"])
        signal_date = str(event["signal_date"])
        if (
            maturity_date not in positions
            or signal_date not in positions
            or positions[maturity_date] - positions[signal_date] != 2
        ):
            return False
        row = forecast.loc[pd.Timestamp(maturity_date)]
        if int(row["matured_on_close_count"]) != 1:
            return False
        for event_field, forecast_field in row_mapping.items():
            left = event[event_field]
            right = row[forecast_field]
            if isinstance(left, (float, np.floating)):
                if float(left) != float(right):
                    return False
            elif left != right:
                return False
    return True


def _integrity_checks(
    *,
    computation: _computation.AuditComputation,
    parent: _parent.ParentBundleEvidence,
    bounded: _input.BoundedAuditSnapshot,
    lock: _control.AttemptLockEvidence,
    git_identity: Mapping[str, Any],
) -> dict[str, bool]:
    scenarios = computation.forks.scenarios
    lead = scenarios[_audit_replay.FROZEN_2023_LEAD]
    shadow = scenarios[_audit_replay.ONLINE_2024_SHADOW]
    if (
        set(computation.integrity_proofs) != _EXPECTED_COMPUTATION_PROOFS
        or not all(
            type(value) is bool
            for value in computation.integrity_proofs.values()
        )
    ):
        raise ContextualExpertAggregation2024AuditError(
            "computation integrity proof inventory changed"
        )
    base = {
        "binary_actions": computation.integrity_proofs["binary_forecast_targets"],
        "bounded_input_identity": (
            bounded.snapshot.raw_sha256 == _input.INPUT_RAW_SHA256
            and bounded.snapshot.spec.canonical_sha256
            == _input.INPUT_CANONICAL_SHA256
            and len(bounded.suffix) == _computation.EXPECTED_SUFFIX_ROWS
        ),
        "contract_branch_pushed_dependency_identity": (
            git_identity["branch"] == _artifacts.EXPECTED_BRANCH
            and git_identity["commit"] == git_identity["upstream_commit"]
            and git_identity["preregistration_is_ancestor"] is True
        ),
        "control_root_and_output_identity": True,
        "durable_attempt_lock_identity": (
            lock.content["git_commit"] == git_identity["commit"]
            and lock.content["artifact_schema_registry_sha256"]
            == _artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
        ),
        "exact_parent_bundle_checkpoint_accounts": parent.verified,
        "exact_parent_model_and_account_fork": computation.integrity_proofs[
            "exact_parent_model_and_account_fork"
        ],
        "exact_prefix_checkpoint_continuity": (
            bounded.prefix_continuity["prefix_market_chain_exact"] is True
            and bounded.prefix_continuity["parent_checkpoint_sha256"]
            == parent.checkpoint.digest_sha256
        ),
        "frozen_lead_zero_post_cutoff_admissions": (
            lead.diagnostics.admitted_events == 0
        ),
        "ledger_episode_xor_gate_reconciliation": all(
            value
            for name, value in computation.integrity_proofs.items()
            if "reconciliation" in name
        ),
        "no_account_reset_or_added_capital": computation.integrity_proofs[
            "no_account_reset_or_added_capital"
        ],
        "no_later_market_data_access": True,
        "no_prior_2024_artifact_access": True,
        "no_short_leverage_borrowing_negative_cash": all(
            value
            for name, value in computation.integrity_proofs.items()
            if name.endswith("__no_leverage")
        ),
        "one_lesson_per_eligible_opportunity": (
            _lesson_conservation(lead, parent_checkpoint=parent.checkpoint)
            and _lesson_conservation(
                shadow, parent_checkpoint=parent.checkpoint
            )
        ),
        "online_shadow_causal_maturity_and_update_order": (
            shadow.checkpoint.model_payload["runtime"]
            == {
                "learning_mode": "causal_online",
                "frozen_cutoff": None,
                "ablation_mode": "full",
            }
            and _causal_update_order(
                shadow,
                parent_checkpoint=parent.checkpoint,
                historical_prefix=bounded.historical_prefix,
            )
        ),
        "pending_and_cooldown_state_exact": all(
            _checkpoint_valid(scenario.checkpoint)
            for scenario in scenarios.values()
        ),
        "sanitized_receipt_identity": (
            _sha256_bytes(bounded.receipt_bytes) == _input.RECEIPT_RAW_SHA256
            and bounded.receipt == _input.RECEIPT_EXPECTED
        ),
        "stage_runtime_protocol_bound": True,
        "zero_external_use": True,
    }
    if set(base) != set(_STATIC_INTEGRITY_CHECKS):
        raise ContextualExpertAggregation2024AuditError(
            "static integrity inventory changed"
        )
    combined = {**base, **dict(computation.integrity_proofs)}
    if not all(type(value) is bool for value in combined.values()):
        raise ContextualExpertAggregation2024AuditError(
            "fatal integrity evidence contains a non-Boolean value"
        )
    return {name: combined[name] for name in sorted(combined)}


def _integrity_payload(
    *,
    checks: Mapping[str, bool],
    parent: _parent.ParentBundleEvidence,
    bounded: _input.BoundedAuditSnapshot,
    lock: _control.AttemptLockEvidence,
    computation: _computation.AuditComputation,
) -> dict[str, Any]:
    unsigned = {
        "integrity_evidence_schema_version": 1,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "evidence_classification": "fatal_integrity_evidence",
        "checks": {name: bool(checks[name]) for name in sorted(checks)},
        "references": {
            "attempt_lock_file_sha256": lock.raw_sha256,
            "parent_manifest_file_sha256": parent.manifest_file_sha256,
            "parent_checkpoint_file_sha256": parent.checkpoint_file_sha256,
            "input_raw_sha256": bounded.snapshot.raw_sha256,
            "input_canonical_sha256": bounded.snapshot.spec.canonical_sha256,
            "parent_checkpoint_sha256": parent.checkpoint.digest_sha256,
            "frozen_action_stream_sha256": computation.action_stream_sha256[
                _audit_replay.FROZEN_2023_LEAD
            ],
            "shadow_action_stream_sha256": computation.action_stream_sha256[
                _audit_replay.ONLINE_2024_SHADOW
            ],
        },
        "zero_use": {
            "network_access": False,
            "news_access": False,
            "llm_calls": 0,
            "api_calls": 0,
            "external_model_calls": 0,
            "external_cost_usd": 0.0,
        },
    }
    return _self_hashed(unsigned, field="integrity_evidence_sha256")


def _runtime_payload(
    *, clock: RuntimeClock, provisional: bool
) -> dict[str, Any]:
    samples = clock.samples or {}
    preseal = samples.get("preseal")
    post_private = samples.get("post_private_verify")
    prepromotion = samples.get("prepromotion")
    if preseal is None:
        raise ContextualExpertAggregation2024AuditError(
            "runtime evidence requires the preseal sample"
        )
    if provisional:
        post_private = None
        prepromotion = None
    elif post_private is None or prepromotion is None:
        raise ContextualExpertAggregation2024AuditError(
            "final runtime evidence requires every frozen sample"
        )
    elif not float(preseal) < float(post_private) < float(prepromotion):
        raise ContextualExpertAggregation2024AuditError(
            "final runtime samples are not strictly increasing"
        )
    unsigned = {
        "runtime_evidence_schema_version": RUNTIME_EVIDENCE_SCHEMA_VERSION,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "clock_name": "time.monotonic",
        "command_start_monotonic": float(clock.start),
        "stage_deadline_seconds": float(_artifacts.STAGE_DEADLINE_SECONDS),
        "finalization_reserve_seconds": float(
            _artifacts.FINALIZATION_RESERVE_SECONDS
        ),
        "samples": {
            "preseal": float(preseal),
            "post_private_verify": (
                None if post_private is None else float(post_private)
            ),
            "prepromotion": None if prepromotion is None else float(prepromotion),
        },
        "stage_internal_deadline_pass": not provisional,
        "final_commit_sample_location": (
            "AUDIT_2024_STAGE_SUCCESS.json.marker_preparation_elapsed_seconds;"
            " bootstrap stdout; preservation evidence"
        ),
        "network_access": False,
        "news_access": False,
        "llm_calls": 0,
        "api_calls": 0,
        "external_model_calls": 0,
        "external_cost_usd": 0.0,
    }
    return _self_hashed(unsigned, field="runtime_evidence_sha256")


def _report_payload(
    *,
    evaluation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    provenance: Mapping[str, Any],
    isolation: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    gates = evaluation["gate_report"]
    stage_pass = bool(gates["stage_pass"])
    unsigned = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "status": gates["status"],
        "stage_pass": stage_pass,
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "is_confirmation": False,
        "is_prospective": False,
        "is_unseen_2024_test": False,
        "post_hoc_frozen_policy_replication": True,
        "historical_results_authorize_real_capital": False,
        "calendar_period": {"first": "2024-01-02", "last": "2024-12-31"},
        "continuous_account_period": {"first_year": 2005, "last_year": 2024},
        "parent_rejection_remains_final": True,
        "later_market_data_accessed": False,
        "prior_2024_artifact_accessed": False,
        "learning_classification": gates["learning_classification"],
        "learning_candidate_for_2025_shadow": gates[
            "learning_candidate_for_2025_shadow"
        ],
        "metrics": _jsonable(evaluation["metrics"]),
        "gate_report": _jsonable(gates),
        "runtime_cost_evidence": _jsonable(runtime),
        "artifact_references": {
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "provenance_sha256": provenance["provenance_sha256"],
            "isolation_evidence_sha256": isolation["isolation_evidence_sha256"],
            "integrity_evidence_sha256": integrity["integrity_evidence_sha256"],
        },
        "next_step": (
            "Preserve this replication; later unseen-period testing remains separately gated."
            if stage_pass
            else "Preserve the rejection; this frozen 2024 attempt is permanently closed."
        ),
    }
    return _self_hashed(unsigned, field="report_sha256")


def _evaluate(
    computation: _computation.AuditComputation,
    *, checks: Mapping[str, bool],
) -> dict[str, Any]:
    policy_evidence = {
        cost: {
            policy: {
                "strategy_ledger": computation.ledgers[cost][policy],
                "cash_episodes": computation.episodes[cost][policy],
            }
            for policy in _artifacts.POLICY_ORDER
        }
        for cost in _artifacts.COST_ORDER
    }
    return _evaluation.evaluate_2024_audit(
        policy_evidence,
        xor_evidence=computation.xors,
        integrity_checks=checks,
    )


def _build_reproducible_payloads(
    *,
    parent: _parent.ParentBundleEvidence,
    bounded: _input.BoundedAuditSnapshot,
    lock: _control.AttemptLockEvidence,
    git_identity: Mapping[str, Any],
    computation: _computation.AuditComputation,
    checks: Mapping[str, bool],
    evaluation: Mapping[str, Any],
    runtime_payload: Mapping[str, Any],
) -> dict[str, bytes]:
    checkpoint = _build_checkpoint(computation)
    provenance = _source_provenance_payload(
        parent=parent, bounded=bounded, lock=lock
    )
    isolation = _known_result_isolation_payload(git_identity=git_identity)
    integrity = _integrity_payload(
        checks=checks,
        parent=parent,
        bounded=bounded,
        lock=lock,
        computation=computation,
    )
    report = _report_payload(
        evaluation=evaluation,
        runtime=runtime_payload,
        checkpoint=checkpoint,
        provenance=provenance,
        isolation=isolation,
        integrity=integrity,
    )
    payloads: dict[str, bytes] = {
        ".gitattributes": _artifacts.GIT_ATTRIBUTES_BYTES,
        _artifacts.ATTEMPT_LOCK_FILENAME: lock.bytes_value,
        "input_snapshot_receipt.json": bounded.receipt_bytes,
        "audit_prices_through_2024.csv": bounded.input_bytes,
        "parent_stage_manifest.json": parent.manifest_bytes,
        "parent_checksums.json": parent.checksums_bytes,
        "parent_verification.json": _json_payload(parent.lock_payload()),
        "prefix_continuity_proof.json": _json_payload(_prefix_payload(bounded)),
        "source_bundle_provenance.json": _json_payload(provenance),
        "known_result_isolation_evidence.json": _json_payload(isolation),
        "audit_fixed_features_2024.table.json": (
            _artifacts.canonical_registered_table_bytes(
                "audit_fixed_features_2024.table.json",
                computation.fixed_features,
            )
        ),
        "audit_forecast__fixed_comparators.table.json": (
            _artifacts.canonical_registered_table_bytes(
                "audit_forecast__fixed_comparators.table.json",
                computation.fixed_comparator_forecast,
            )
        ),
        "audit_state_weight_diagnostics_2024.table.json": (
            _artifacts.canonical_registered_table_bytes(
                "audit_state_weight_diagnostics_2024.table.json",
                computation.state_weight_diagnostics,
            )
        ),
        "audit_continuation_checkpoint_through_2024.json": _json_payload(
            checkpoint
        ),
        "audit_pending_lessons.json": _json_payload(
            _pending_payload(computation)
        ),
        "audit_replay_diagnostics.json": _json_payload(
            _replay_payload(computation)
        ),
        "audit_metrics.json": _json_payload(_metrics_payload(evaluation)),
        "audit_gate_report.json": _json_payload(_gate_payload(evaluation)),
        "audit_integrity_evidence.json": _json_payload(integrity),
        "audit_runtime_cost_evidence.json": _json_payload(runtime_payload),
        "report.json": _json_payload(report),
    }
    for scenario in _artifacts.SCENARIO_ORDER:
        payloads[f"audit_forecast__{scenario}.table.json"] = (
            _artifacts.canonical_registered_table_bytes(
                f"audit_forecast__{scenario}.table.json",
                computation.forecasts[scenario],
            )
        )
        payloads[f"audit_matured_lessons__{scenario}.table.json"] = (
            _artifacts.canonical_registered_table_bytes(
                f"audit_matured_lessons__{scenario}.table.json",
                computation.matured_lessons[scenario],
            )
        )
    for cost in _artifacts.COST_ORDER:
        for policy in _artifacts.POLICY_ORDER:
            filename = f"audit_ledger__{cost}__{policy}.table.json"
            payloads[filename] = _artifacts.canonical_registered_table_bytes(
                filename, computation.ledgers[cost][policy]
            )
        payloads[f"audit_episodes__{cost}.json"] = _json_payload(
            _episodes_payload(computation, cost=cost)
        )
        payloads[f"audit_xor__{cost}.json"] = _json_payload(
            _xor_payload(computation, cost=cost)
        )
    return _artifacts.validate_payload_inventory(payloads)


def _manifest_fields(
    *,
    git_identity: Mapping[str, Any],
    parent: _parent.ParentBundleEvidence,
    bounded: _input.BoundedAuditSnapshot,
    lock: _control.AttemptLockEvidence,
    evaluation: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    gates = evaluation["gate_report"]
    return {
        "manifest_schema_version": _artifacts.MANIFEST_SCHEMA_VERSION,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "verifier_id": _artifacts.VERIFIER_ID,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "status": gates["status"],
        "stage_pass": bool(gates["stage_pass"]),
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "preregistration_commit": _artifacts.PREREGISTRATION_COMMIT,
        "git_identity": _jsonable(dict(git_identity)),
        "dependency_identity_sha256": git_identity[
            "dependency_identity_sha256"
        ],
        "artifact_schema_registry_sha256": (
            _artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
        ),
        "parent_manifest_file_sha256": parent.manifest_file_sha256,
        "parent_manifest_self_sha256": parent.manifest_self_sha256,
        "parent_checkpoint_file_sha256": parent.checkpoint_file_sha256,
        "parent_checkpoint_self_sha256": parent.checkpoint_self_sha256,
        "receipt_file_sha256": _sha256_bytes(bounded.receipt_bytes),
        "input_raw_sha256": bounded.snapshot.raw_sha256,
        "input_canonical_sha256": bounded.snapshot.spec.canonical_sha256,
        "attempt_lock_file_sha256": lock.raw_sha256,
        "runtime_cost_evidence_file_sha256": _sha256_bytes(
            payloads["audit_runtime_cost_evidence.json"]
        ),
        "gate_report_file_sha256": _sha256_bytes(
            payloads["audit_gate_report.json"]
        ),
        "learning_classification": gates["learning_classification"],
        "later_market_data_accessed": False,
        "prior_2024_artifact_accessed": False,
        "external_cost_usd": 0.0,
    }


def reconstruct_reproducible_payloads(
    repo_root: Path,
    directory: Path,
    *,
    lock_bytes: bytes,
    receipt_bytes: bytes,
    input_bytes: bytes,
    runtime_payload: Mapping[str, Any],
) -> dict[str, bytes]:
    """Independently rebuild the exact 41 payload bytes from sealed primitives."""

    root = Path(repo_root)
    bundle = Path(directory)
    try:
        lock_value = _artifacts.parse_canonical_json_line(
            lock_bytes, field="sealed attempt lock"
        )
    except _artifacts.ContextualExpertAggregation2024AuditArtifactError as exc:
        raise ContextualExpertAggregation2024AuditError(
            "sealed attempt lock is not canonical"
        ) from exc
    if not isinstance(lock_value, Mapping):
        raise ContextualExpertAggregation2024AuditError(
            "sealed attempt lock is not an object"
        )
    lock = _control.AttemptLockEvidence(
        path=bundle / _artifacts.ATTEMPT_LOCK_FILENAME,
        content=dict(lock_value),
        bytes_value=lock_bytes,
        raw_sha256=_sha256_bytes(lock_bytes),
    )
    parent = _parent.verify_parent_bundle(root)
    if lock.content.get("parent_verification") != parent.lock_payload():
        raise ContextualExpertAggregation2024AuditError(
            "sealed lock parent identity differs from regeneration"
        )
    bounded = _input.load_postlock_bounded_snapshot(
        input_path=bundle / "audit_prices_through_2024.csv",
        receipt_path=bundle / "input_snapshot_receipt.json",
        checkpoint=parent.checkpoint,
        verified_receipt_bytes=receipt_bytes,
        verified_input_bytes=input_bytes,
    )
    computation = _computation.compute_2024_continuation(
        parent=parent, bounded=bounded
    )
    git_identity = lock.content.get("git_identity")
    if not isinstance(git_identity, Mapping):
        raise ContextualExpertAggregation2024AuditError(
            "sealed lock lacks its frozen Git identity"
        )
    checks = _integrity_checks(
        computation=computation,
        parent=parent,
        bounded=bounded,
        lock=lock,
        git_identity=git_identity,
    )
    evaluation = _evaluate(computation, checks=checks)
    return _build_reproducible_payloads(
        parent=parent,
        bounded=bounded,
        lock=lock,
        git_identity=git_identity,
        computation=computation,
        checks=checks,
        evaluation=evaluation,
        runtime_payload=runtime_payload,
    )


def _assert_control_inventory(
    repo_root: Path,
    *,
    run_entries: Sequence[str],
    marker: str | None = None,
) -> None:
    root_entries = [
        _artifacts.ATTEMPT_LOCK_FILENAME,
        "authorized_inputs",
        "input_snapshot_receipt.json",
        "runs",
    ]
    if marker is not None:
        root_entries.append(marker)
    expected = {
        "control_root": sorted(root_entries),
        "authorized_inputs": ["aapl_spy_qqq_through_2024.csv"],
        "runs": sorted(run_entries),
    }
    if _control.control_inventory(repo_root) != expected:
        raise ContextualExpertAggregation2024AuditError(
            "control-root inventory differs from the frozen transition"
        )


def _ordinary_file(path: Path) -> None:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise ContextualExpertAggregation2024AuditError(
            "sealed file is missing or unreadable"
        ) from exc
    if (
        not stat.S_ISREG(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or bool(getattr(value, "st_file_attributes", 0) & 0x0400)
    ):
        raise ContextualExpertAggregation2024AuditError(
            "sealed path is not an ordinary file"
        )


def _assert_exact_bundle_bytes(
    directory: Path,
    *,
    payloads: Mapping[str, bytes],
    metadata: _artifacts.BundleMetadata,
) -> None:
    try:
        entries = {entry.name for entry in os.scandir(directory)}
    except OSError as exc:
        raise ContextualExpertAggregation2024AuditError(
            "private bundle inventory is unreadable"
        ) from exc
    if entries != _artifacts.BUNDLE_FILENAMES:
        raise ContextualExpertAggregation2024AuditError(
            "private bundle inventory changed"
        )
    expected = {
        **dict(payloads),
        "checksums.json": metadata.checksums_bytes,
        "stage_manifest.json": metadata.manifest_bytes,
    }
    for filename in _artifacts.BUNDLE_FILE_ORDER:
        path = directory / filename
        _ordinary_file(path)
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise ContextualExpertAggregation2024AuditError(
                "private bundle file became unreadable"
            ) from exc
        if observed != expected[filename]:
            raise ContextualExpertAggregation2024AuditError(
                "private bundle bytes differ from deterministic construction"
            )
    _artifacts.parse_checksums_bytes(metadata.checksums_bytes)
    _artifacts.parse_stage_manifest_bytes(metadata.manifest_bytes)


def _atomic_replace_file(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.rewrite")
    if temporary.exists():
        raise ContextualExpertAggregation2024AuditError(
            "private rewrite temporary path already exists"
        )
    _experiment._exclusive_write(temporary, payload)
    try:
        os.replace(temporary, path)
        _experiment._fsync_directory(path.parent)
    except OSError as exc:
        raise ContextualExpertAggregation2024AuditError(
            "atomic private bundle rewrite failed"
        ) from exc


def _rewrite_private_runtime(
    directory: Path,
    *,
    provisional_payloads: Mapping[str, bytes],
    final_payloads: Mapping[str, bytes],
    metadata: _artifacts.BundleMetadata,
) -> None:
    changed = {
        name
        for name in _artifacts.PAYLOAD_FILE_ORDER
        if provisional_payloads[name] != final_payloads[name]
    }
    if changed != {"audit_runtime_cost_evidence.json", "report.json"}:
        raise ContextualExpertAggregation2024AuditError(
            "final runtime rewrite changed an unauthorized economic payload"
        )
    _atomic_replace_file(
        directory / "audit_runtime_cost_evidence.json",
        final_payloads["audit_runtime_cost_evidence.json"],
    )
    _atomic_replace_file(directory / "report.json", final_payloads["report.json"])
    _atomic_replace_file(directory / "stage_manifest.json", metadata.manifest_bytes)
    _atomic_replace_file(directory / "checksums.json", metadata.checksums_bytes)
    _assert_exact_bundle_bytes(
        directory, payloads=final_payloads, metadata=metadata
    )


def _exclusive_marker(path: Path, payload: bytes) -> None:
    _experiment._exclusive_write(path, payload)
    _experiment._fsync_directory(path.parent)
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise ContextualExpertAggregation2024AuditError(
            "pending success marker is unreadable"
        ) from exc
    if observed != payload:
        raise ContextualExpertAggregation2024AuditError(
            "pending success marker changed during readback"
        )
    _artifacts.parse_success_marker_bytes(observed)


def _failure_transition(
    repo_root: Path, *, attempt_locked: bool, marker_committed: bool
) -> None:
    if not attempt_locked or marker_committed:
        return
    root = Path(repo_root)
    pending = root / _artifacts.PENDING_SUCCESS_MARKER_PATH
    success = root / _artifacts.SUCCESS_MARKER_PATH
    private = root / _artifacts.PRIVATE_DIRECTORY
    final = root / _artifacts.OUTPUT_DIRECTORY
    failed = root / _artifacts.FAILED_DIRECTORY
    try:
        for marker in (pending, success):
            if marker.exists():
                _ordinary_file(marker)
                marker.unlink()
        if pending.parent.exists():
            _experiment._fsync_directory(pending.parent)
        candidates = [path for path in (private, final) if path.exists()]
        if candidates:
            if len(candidates) != 1 or failed.exists():
                raise ContextualExpertAggregation2024AuditError(
                    "forensic failure transition has ambiguous paths"
                )
            os.replace(candidates[0], failed)
            _experiment._fsync_directory(failed.parent)
            _control.harden_runtime_paths(root)
            _assert_control_inventory(root, run_entries=[failed.name])
    except Exception as exc:
        raise ContextualExpertAggregation2024AuditError(
            "mandatory forensic failure preservation failed"
        ) from exc


def _run_audit(
    repo_root: Path,
    *,
    bootstrap_attestation: Mapping[str, Any],
    clock: RuntimeClock,
) -> dict[str, Any]:
    root = Path(repo_root)
    attempt_locked = False
    marker_committed = False
    try:
        clock.require(limit=_artifacts.STAGE_DEADLINE_SECONDS)
        _control.harden_runtime_paths(root)
        git_identity = _control.attest_prelock_git(root)
        parent = _parent.verify_parent_bundle(root)
        repeated_git = _control.attest_prelock_git(root)
        if repeated_git != git_identity:
            raise ContextualExpertAggregation2024AuditError(
                "Git identity changed during parent verification"
            )
        pristine = _control.require_pristine_control_root(root)
        clock.require(limit=_artifacts.STAGE_DEADLINE_SECONDS)
        lock = _control.create_attempt_lock(
            repo_root=root,
            git_identity=git_identity,
            parent_verification=parent.lock_payload(),
            pristine_inventory=pristine,
        )
        attempt_locked = True
        _control.harden_runtime_paths(root)
        _control.reattest_after_lock(
            root,
            expected_git_identity=git_identity,
            expected_lock=lock,
        )
        repeated_parent = _parent.verify_parent_bundle(root)
        if repeated_parent.lock_payload() != parent.lock_payload():
            raise ContextualExpertAggregation2024AuditError(
                "parent identity changed after the attempt lock"
            )
        _control.harden_runtime_paths(root)
        receipt_bytes = _control.read_postlock_tracked_file(
            root,
            relative_path=_input.RECEIPT_PATH,
            expected_blob=_input.RECEIPT_GIT_BLOB,
            expected_raw_sha256=_input.RECEIPT_RAW_SHA256,
        )
        _input.parse_sanitized_receipt_bytes(receipt_bytes)
        _control.harden_runtime_paths(root)
        input_bytes = _control.read_postlock_tracked_file(
            root,
            relative_path=_input.INPUT_PATH,
            expected_blob=_input.INPUT_GIT_BLOB,
            expected_raw_sha256=_input.INPUT_RAW_SHA256,
        )
        bounded = _input.load_postlock_bounded_snapshot(
            input_path=root / _input.INPUT_PATH,
            receipt_path=root / _input.RECEIPT_PATH,
            checkpoint=parent.checkpoint,
            verified_receipt_bytes=receipt_bytes,
            verified_input_bytes=input_bytes,
        )
        computation = _computation.compute_2024_continuation(
            parent=parent, bounded=bounded
        )
        checks = _integrity_checks(
            computation=computation,
            parent=parent,
            bounded=bounded,
            lock=lock,
            git_identity=git_identity,
        )
        evaluation = _evaluate(computation, checks=checks)
        clock.sample("preseal", limit=_artifacts.STAGE_DEADLINE_SECONDS)
        provisional_runtime = _runtime_payload(clock=clock, provisional=True)
        provisional_payloads = _build_reproducible_payloads(
            parent=parent,
            bounded=bounded,
            lock=lock,
            git_identity=git_identity,
            computation=computation,
            checks=checks,
            evaluation=evaluation,
            runtime_payload=provisional_runtime,
        )
        provisional_metadata = _artifacts.build_bundle_metadata(
            provisional_payloads,
            manifest_fields=_manifest_fields(
                git_identity=git_identity,
                parent=parent,
                bounded=bounded,
                lock=lock,
                evaluation=evaluation,
                payloads=provisional_payloads,
            ),
        )
        _control.harden_runtime_paths(root)
        runs = root / _artifacts.OUTPUT_PARENT
        runs.mkdir(parents=False, exist_ok=False)
        _experiment._fsync_directory(runs.parent)
        _control.harden_runtime_paths(root)
        _assert_control_inventory(root, run_entries=[])
        private = root / _artifacts.PRIVATE_DIRECTORY
        _artifacts.write_private_bundle(
            private,
            payloads=provisional_payloads,
            metadata=provisional_metadata,
        )
        _assert_control_inventory(root, run_entries=[private.name])
        from .contextual_expert_aggregation_2024_audit_verifier import (
            verify_private_audit_bundle,
        )

        verify_private_audit_bundle(
            private,
            repo_root=root,
            expected_manifest_sha256=provisional_metadata.manifest[
                "manifest_sha256"
            ],
            allow_provisional_runtime=True,
        )
        clock.sample("post_private_verify", limit=_artifacts.STAGE_DEADLINE_SECONDS)
        clock.sample("prepromotion", limit=PREPROMOTION_DEADLINE_SECONDS)
        final_runtime = _runtime_payload(clock=clock, provisional=False)
        final_payloads = _build_reproducible_payloads(
            parent=parent,
            bounded=bounded,
            lock=lock,
            git_identity=git_identity,
            computation=computation,
            checks=checks,
            evaluation=evaluation,
            runtime_payload=final_runtime,
        )
        final_metadata = _artifacts.build_bundle_metadata(
            final_payloads,
            manifest_fields=_manifest_fields(
                git_identity=git_identity,
                parent=parent,
                bounded=bounded,
                lock=lock,
                evaluation=evaluation,
                payloads=final_payloads,
            ),
        )
        _rewrite_private_runtime(
            private,
            provisional_payloads=provisional_payloads,
            final_payloads=final_payloads,
            metadata=final_metadata,
        )
        verify_private_audit_bundle(
            private,
            repo_root=root,
            expected_manifest_sha256=final_metadata.manifest[
                "manifest_sha256"
            ],
            allow_provisional_runtime=False,
        )
        live_prepromotion_guard = clock.require(
            limit=PREPROMOTION_DEADLINE_SECONDS
        )
        _control.harden_runtime_paths(root)
        _assert_control_inventory(root, run_entries=[private.name])
        final = root / _artifacts.OUTPUT_DIRECTORY
        os.replace(private, final)
        _experiment._fsync_directory(runs)
        _control.harden_runtime_paths(root)
        _assert_control_inventory(root, run_entries=[final.name])
        _assert_exact_bundle_bytes(
            final, payloads=final_payloads, metadata=final_metadata
        )
        marker_preparation = clock.elapsed()
        if (
            marker_preparation <= live_prepromotion_guard
            or not marker_preparation < _artifacts.STAGE_DEADLINE_SECONDS
        ):
            raise ContextualExpertAggregation2024AuditError(
                "success-marker preparation time is out of order or over deadline"
            )
        marker = _artifacts.build_success_marker(
            {
                "marker_schema_version": _artifacts.MARKER_SCHEMA_VERSION,
                "contract_version": _artifacts.CONTRACT_VERSION,
                "run_id": _artifacts.AUDIT_RUN_ID,
                "stage": _artifacts.AUDIT_STAGE,
                "preregistration_commit": _artifacts.PREREGISTRATION_COMMIT,
                "git_commit": git_identity["commit"],
                "attempt_lock_file_sha256": lock.raw_sha256,
                "final_relative_path": _artifacts.OUTPUT_DIRECTORY.as_posix(),
                "final_inventory_sha256": _artifacts.FINAL_INVENTORY_SHA256,
                "stage_manifest_file_sha256": _sha256_bytes(
                    final_metadata.manifest_bytes
                ),
                "stage_manifest_self_sha256": final_metadata.manifest[
                    "manifest_sha256"
                ],
                "checksums_file_sha256": _sha256_bytes(
                    final_metadata.checksums_bytes
                ),
                "marker_preparation_elapsed_seconds": float(marker_preparation),
                "stage_deadline_seconds": float(
                    _artifacts.STAGE_DEADLINE_SECONDS
                ),
                "marker_preparation_deadline_pass": True,
                "external_cost_usd": 0.0,
            }
        )
        marker_bytes = _artifacts.success_marker_bytes(marker)
        pending = root / _artifacts.PENDING_SUCCESS_MARKER_PATH
        _exclusive_marker(pending, marker_bytes)
        _assert_control_inventory(
            root,
            run_entries=[final.name],
            marker=_artifacts.PENDING_SUCCESS_MARKER_FILENAME,
        )
        final_commit_sample = clock.sample(
            "final_commit_sample", limit=_artifacts.STAGE_DEADLINE_SECONDS
        )
        result = {
            "contract_version": _artifacts.CONTRACT_VERSION,
            "run_id": _artifacts.AUDIT_RUN_ID,
            "stage": _artifacts.AUDIT_STAGE,
            "status": evaluation["gate_report"]["status"],
            "stage_pass": bool(evaluation["gate_report"]["stage_pass"]),
            "learning_classification": evaluation["gate_report"][
                "learning_classification"
            ],
            "learning_candidate_for_2025_shadow": evaluation["gate_report"][
                "learning_candidate_for_2025_shadow"
            ],
            "output_directory": _artifacts.OUTPUT_DIRECTORY.as_posix(),
            "manifest_sha256": final_metadata.manifest["manifest_sha256"],
            "success_marker_sha256": marker["marker_sha256"],
            "stage_runtime_samples": {
                **dict(clock.samples or {}),
                "live_prepromotion_guard": live_prepromotion_guard,
                "final_commit_sample": final_commit_sample,
            },
            "external_cost_usd": 0.0,
        }
        success = root / _artifacts.SUCCESS_MARKER_PATH
        os.replace(pending, success)
        try:
            _experiment._fsync_directory(success.parent)
        except Exception:
            if success.exists():
                success.unlink()
                _experiment._fsync_directory(success.parent)
            raise
        marker_committed = True
        return result
    except Exception:
        _failure_transition(
            root,
            attempt_locked=attempt_locked,
            marker_committed=marker_committed,
        )
        raise


def run_stage(stage: str) -> dict[str, Any]:
    """Execute the sole authorized 2024 audit stage under isolated bootstrap."""

    if stage != _artifacts.AUDIT_STAGE:
        raise ContextualExpertAggregation2024AuditError(
            "the 2024 audit runner accepts only audit_2024"
        )
    from .contextual_expert_aggregation_2024_audit_bootstrap import (
        require_active_attestation,
    )

    root = Path(__file__).resolve().parents[1]
    attestation = require_active_attestation("stage", stage, root)
    start = attestation.get("monotonic_start")
    if type(start) is not float:
        raise ContextualExpertAggregation2024AuditError(
            "bootstrap attestation lacks its monotonic start"
        )
    return _run_audit(
        root,
        bootstrap_attestation=attestation,
        clock=RuntimeClock(start=start),
    )


__all__ = [
    "ContextualExpertAggregation2024AuditError",
    "RuntimeClock",
    "StageContext",
    "reconstruct_reproducible_payloads",
    "run_stage",
]
