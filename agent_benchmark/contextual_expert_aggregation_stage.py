"""Frozen stage runner for causal contextual expert aggregation.

The command surface is deliberately tiny::

    python -m agent_benchmark.contextual_expert_aggregation_stage development
    python -m agent_benchmark.contextual_expert_aggregation_stage confirmation

Inputs, outputs, run identifiers, costs, policies, cutoffs, and retry semantics
are source-frozen.  This module performs no network, news, API, or LLM work.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import contextual_expert_aggregation_artifacts as _artifacts
from . import contextual_expert_aggregation_evaluation as _evaluation
from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_ledger as _ledger
from . import contextual_expert_aggregation_replay as _replay
from .contextual_expert_aggregation import (
    EXPERT_NAMES,
    FAST_LOOKBACK,
    LESSON_MEMORY_START_ISO,
    LONG_TIE_TOLERANCE,
    MARKET_STATE_NAMES,
    MODEL_METHOD,
    PER_SIDE_FRICTION,
    SCALE_NAMES,
    SERIALIZATION_VERSION,
    SLOW_LOOKBACK,
    STATE_PRIOR_STRENGTH,
)


CONTRACT_VERSION = _artifacts.CONTRACT_VERSION
DEVELOPMENT_STAGE = _artifacts.DEVELOPMENT_STAGE
CONFIRMATION_STAGE = _artifacts.CONFIRMATION_STAGE
STAGE_ORDER = _artifacts.STAGE_ORDER
ARM_ORDER = _artifacts.ARM_ORDER
FIXED_POLICY_ORDER = _artifacts.FIXED_POLICY_ORDER
POLICY_ORDER = _artifacts.POLICY_ORDER
COST_ORDER = _artifacts.COST_ORDER
COST_BPS = _artifacts.COST_BPS
RUN_ID_BY_STAGE = _artifacts.RUN_ID_BY_STAGE
OUTPUT_DIRECTORY_BY_STAGE = _artifacts.OUTPUT_DIRECTORY_BY_STAGE
OUTPUT_PARENT = _artifacts.OUTPUT_PARENT
RUN_TIME_LIMIT_SECONDS = _experiment.RUN_TIME_LIMIT_SECONDS

ONLINE_FULL_ARM = _replay.ONLINE_FULL_ARM
FROZEN_2018_ARM = _replay.FROZEN_2018_ARM
GLOBAL_ONLY_ARM = _replay.GLOBAL_ONLY_ARM
LIFETIME_ONLY_ARM = _replay.LIFETIME_ONLY_ARM

DEVELOPMENT_MANIFEST_PATH = (
    OUTPUT_DIRECTORY_BY_STAGE[DEVELOPMENT_STAGE] / "stage_manifest.json"
)
FROZEN_COMMAND_BY_STAGE: Mapping[str, tuple[str, ...]] = {
    stage: (
        "python",
        "-m",
        "agent_benchmark.contextual_expert_aggregation_stage",
        stage,
    )
    for stage in STAGE_ORDER
}

PARENT_FORECAST_BY_STAGE: Mapping[str, str] = {
    DEVELOPMENT_STAGE: "development_forecast_through_2018.csv",
    CONFIRMATION_STAGE: "validation_online_forecast.csv",
}
PARENT_TARGET_COLUMN_MAP: Mapping[str, str] = {
    "fixed_always_long_target_exposure": "always_long_target_exposure",
    "fixed_union_cash_target_exposure": "unfiltered_union_target_exposure",
    "fixed_contextual_only_target_exposure": (
        "unfiltered_contextual_target_exposure"
    ),
    "fixed_weak_trend_only_target_exposure": (
        "unfiltered_weak_trend_target_exposure"
    ),
}

ABLATION_POLICY_BY_COMPARISON: Mapping[str, str] = {
    "online_minus_frozen_2018": FROZEN_2018_ARM,
    "full_minus_global_only": GLOBAL_ONLY_ARM,
    "full_minus_lifetime_only": LIFETIME_ONLY_ARM,
}

FATAL_GATE_SCHEMA_VERSION = 1
STAGE_EVIDENCE_SCHEMA_VERSION = 1
NO_EXECUTED_TERMINAL_CASH = "no_executed_terminal_cash_residual"
NO_PARTIAL_XOR_BOUNDARY = "no_partial_xor_boundary"


class ContextualExpertAggregationStageError(RuntimeError):
    """Raised when a frozen stage invariant is violated."""


def _fixed_feature_types() -> tuple[str, ...]:
    float_columns = {
        *_replay.CANONICAL_MARKET_COLUMNS,
        "aapl_intraday_return",
        *_replay.COMPARATOR_TARGET_COLUMNS,
    }
    nullable_float_columns = {
        "contextual_prior_intraday_percentile",
        "contextual_spy_return_10",
        "contextual_qqq_return_10",
        "weak_trend_prior_intraday_percentile",
        "weak_trend_spy_return_20",
        "weak_trend_qqq_return_20",
        "weak_trend_aapl_sma_20",
    }
    return tuple(
        (
            "float"
            if name in float_columns
            else "nullable_float"
            if name in nullable_float_columns
            else "bool"
        )
        for name in _replay.FIXED_FEATURE_COLUMNS
    )


FIXED_FEATURE_SCHEMA = _artifacts.CanonicalTableSchema(
    columns=tuple(_replay.FIXED_FEATURE_COLUMNS),
    column_types=_fixed_feature_types(),
    index_name="decision_date",
    index_type="iso_date",
)

FIXED_COMPARATOR_SCHEMA = _artifacts.CanonicalTableSchema(
    columns=tuple(_replay.COMPARATOR_TARGET_COLUMNS),
    column_types=tuple("float" for _ in _replay.COMPARATOR_TARGET_COLUMNS),
    index_name="decision_date",
    index_type="iso_date",
)


def _forecast_types() -> tuple[str, ...]:
    boolean = {
        "contextual_virtual_signal",
        "weak_trend_virtual_signal",
        "unfiltered_union_candidate_signal",
        "canonical_union_opportunity",
        "matured_admitted",
        *(f"advice__{name}" for name in EXPERT_NAMES),
        *(f"matured_advice__{name}" for name in EXPERT_NAMES),
        *(f"scale_active__{scale}" for scale in SCALE_NAMES),
    }
    integer = {
        "matured_on_close_count",
        "processed_session_count",
        "matured_lesson_count",
        "admitted_lesson_count",
        "pending_lesson_count",
        "opportunity_count_to_date",
    }
    strings = {"market_state", "action", "matured_signal_date", "matured_signal_market_state"}
    nullable_float = {
        "matured_entry_adjusted_open",
        "matured_exit_adjusted_open",
        "matured_net_cash_log_edge_10bps",
        "matured_reward_range",
        *(f"matured_reward__{name}" for name in EXPERT_NAMES),
        *(f"weight__{scale}__{name}" for scale in SCALE_NAMES for name in EXPERT_NAMES),
        *(f"rho__{scale}" for scale in SCALE_NAMES),
    }
    result: list[str] = []
    for name in _replay.FORECAST_COLUMNS:
        if name in boolean:
            result.append("bool")
        elif name in integer:
            result.append("int")
        elif name in strings:
            result.append("string")
        elif name in nullable_float:
            result.append("nullable_float")
        else:
            result.append("float")
    return tuple(result)


FORECAST_SCHEMA = _artifacts.CanonicalTableSchema(
    columns=tuple(_replay.FORECAST_COLUMNS),
    column_types=_forecast_types(),
    index_name="decision_date",
    index_type="iso_date",
)


def _matured_types() -> tuple[str, ...]:
    result: list[str] = []
    for name in _replay.MATURED_EVENT_COLUMNS:
        if name in ("signal_date", "maturity_date"):
            result.append("iso_date")
        elif name == "signal_market_state":
            result.append("string")
        elif name in _replay.MATURED_EVENT_BOOL_COLUMNS:
            result.append("bool")
        else:
            result.append("float")
    return tuple(result)


MATURED_EVENT_SCHEMA = _artifacts.CanonicalTableSchema(
    columns=tuple(_replay.MATURED_EVENT_COLUMNS),
    column_types=_matured_types(),
)


def _ledger_types() -> tuple[str, ...]:
    integer = {
        "row_index",
        "held_exposure_before_fill",
        "requested_target_exposure",
        "post_fill_exposure",
        "close_decision_target_exposure",
    }
    boolean = {"inception_fill", "target_changed", "trade_executed"}
    strings = {
        "policy_name",
        "decision_date",
        "fill_date",
        "transition",
        "previous_row_sha256",
        "row_sha256",
    }
    result: list[str] = []
    for name in _ledger.LEDGER_COLUMNS:
        if name == "prior_requested_target_exposure":
            result.append("nullable_int")
        elif name in integer:
            result.append("int")
        elif name in boolean:
            result.append("bool")
        elif name in strings:
            result.append("string")
        else:
            result.append("float")
    return tuple(result)


LEDGER_SCHEMA = _artifacts.CanonicalTableSchema(
    columns=tuple(_ledger.LEDGER_COLUMNS),
    column_types=_ledger_types(),
)

DIAGNOSTIC_COLUMNS = (
    "arm",
    "decision_date",
    "market_state",
    "Q",
    "cash_score",
    "action",
    "learner_target_exposure",
    *(f"aggregate_weight__{name}" for name in EXPERT_NAMES),
    *(f"scale_active__{scale}" for scale in SCALE_NAMES),
    *(f"rho__{scale}" for scale in SCALE_NAMES),
    "processed_session_count",
    "matured_lesson_count",
    "admitted_lesson_count",
    "pending_lesson_count",
    "opportunity_count_to_date",
)


def _diagnostic_types() -> tuple[str, ...]:
    result: list[str] = []
    for name in DIAGNOSTIC_COLUMNS:
        if name in {"arm", "market_state", "action"}:
            result.append("string")
        elif name == "decision_date":
            result.append("iso_date")
        elif name.startswith("scale_active__"):
            result.append("bool")
        elif name.startswith("rho__"):
            result.append("nullable_float")
        elif name.endswith("_count") or name.endswith("_count_to_date"):
            result.append("int")
        else:
            result.append("float")
    return tuple(result)


DIAGNOSTIC_SCHEMA = _artifacts.CanonicalTableSchema(
    columns=DIAGNOSTIC_COLUMNS,
    column_types=_diagnostic_types(),
)


MODEL_CONSTANTS: Mapping[str, Any] = {
    "model_method": MODEL_METHOD,
    "serialization_version": SERIALIZATION_VERSION,
    "expert_order": list(EXPERT_NAMES),
    "scale_order": list(SCALE_NAMES),
    "market_state_order": list(MARKET_STATE_NAMES),
    "fast_lookback": FAST_LOOKBACK,
    "slow_lookback": SLOW_LOOKBACK,
    "lesson_memory_start": LESSON_MEMORY_START_ISO,
    "state_prior_strength": STATE_PRIOR_STRENGTH,
    "per_side_friction": PER_SIDE_FRICTION,
    "long_tie_tolerance": LONG_TIE_TOLERANCE,
    "replay_schema_version": _replay.REPLAY_SCHEMA_VERSION,
    "feature_lookback_sessions": _replay.FEATURE_LOOKBACK_SESSIONS,
}


def _stage(value: Any) -> str:
    if type(value) is not str or value not in STAGE_ORDER:
        raise ContextualExpertAggregationStageError(
            "stage must be exactly development or confirmation"
        )
    return value


def _jsonable(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            raise ContextualExpertAggregationStageError(
                "stage evidence may not contain nonfinite JSON numbers"
            )
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        if not math.isfinite(result):
            raise ContextualExpertAggregationStageError(
                "stage evidence may not contain nonfinite JSON numbers"
            )
        return result
    if isinstance(value, pd.Timestamp):
        if value.tz is not None or value.time() != pd.Timestamp(value.date()).time():
            raise ContextualExpertAggregationStageError(
                "stage evidence timestamp is not a naive session date"
            )
        return value.date().isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise ContextualExpertAggregationStageError(
        f"stage evidence contains unsupported value type: {type(value).__name__}"
    )


def _json_bytes(value: Any) -> bytes:
    return _experiment.pretty_json_bytes(_jsonable(value))


def _strict_json(payload: bytes, *, field: str) -> Any:
    if type(payload) is not bytes:
        raise ContextualExpertAggregationStageError(f"{field} must be exact bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContextualExpertAggregationStageError(
            f"{field} is not strict JSON"
        ) from exc
    if _json_bytes(value) != payload:
        raise ContextualExpertAggregationStageError(
            f"{field} is not canonical pretty JSON"
        )
    return value


def _sha256_json(value: Any) -> str:
    return _experiment.sha256_bytes(_experiment.canonical_json_bytes(_jsonable(value)))


def _normalize_nullable_frame(
    frame: pd.DataFrame, schema: _artifacts.CanonicalTableSchema
) -> pd.DataFrame:
    if tuple(frame.columns) != schema.columns:
        raise ContextualExpertAggregationStageError(
            "frame does not match its frozen artifact schema"
        )
    result = frame.copy().astype(object)
    for name, scalar_type in zip(schema.columns, schema.column_types):
        if scalar_type.startswith("nullable_"):
            values = [
                None
                if value is None
                or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))
                else value
                for value in result[name].tolist()
            ]
            result[name] = pd.Series(
                values,
                index=result.index,
                dtype=object,
            )
    return result


def table_bytes(
    frame: pd.DataFrame, *, schema: _artifacts.CanonicalTableSchema
) -> bytes:
    return _artifacts.canonical_table_bytes(
        _normalize_nullable_frame(frame, schema), schema=schema
    )


def _decision_table_bytes(
    frame: pd.DataFrame, *, schema: _artifacts.CanonicalTableSchema
) -> bytes:
    if schema.index_name != "decision_date":
        raise ContextualExpertAggregationStageError(
            "decision-table helper requires the frozen decision-date index"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ContextualExpertAggregationStageError(
            "decision table requires a DatetimeIndex"
        )
    return table_bytes(frame.rename_axis("decision_date"), schema=schema)


def ledger_table_bytes(frame: pd.DataFrame) -> bytes:
    if tuple(frame.columns) != _ledger.LEDGER_COLUMNS:
        raise ContextualExpertAggregationStageError(
            "ledger artifact does not have the canonical ledger columns"
        )
    normalized = frame.copy().astype(object)
    normalized["prior_requested_target_exposure"] = pd.Series(
        [
            None if value == "" else value
            for value in normalized["prior_requested_target_exposure"].tolist()
        ],
        index=normalized.index,
        dtype=object,
    )
    return table_bytes(normalized, schema=LEDGER_SCHEMA)


def parse_ledger_table_bytes(payload: bytes) -> pd.DataFrame:
    frame = _artifacts.parse_canonical_table_bytes(payload, schema=LEDGER_SCHEMA)
    frame["prior_requested_target_exposure"] = frame[
        "prior_requested_target_exposure"
    ].map(lambda value: "" if value is None else value)
    return frame.loc[:, list(_ledger.LEDGER_COLUMNS)]


@dataclass
class _RuntimeTrace:
    deadline: _experiment.StageDeadline
    phases: list[dict[str, Any]]

    @classmethod
    def start(cls, clock: Callable[[], float]) -> "_RuntimeTrace":
        return cls(
            deadline=_experiment.StageDeadline(clock),
            phases=[],
        )

    def mark(self, phase: str) -> float:
        if type(phase) is not str or not phase or any(
            item["phase"] == phase for item in self.phases
        ):
            raise ContextualExpertAggregationStageError(
                "runtime phase names must be unique nonempty strings"
            )
        elapsed = self.deadline.check(phase)
        self.phases.append({"phase": phase, "elapsed_seconds": float(elapsed)})
        return float(elapsed)

    def evidence(self, *, stage: str) -> dict[str, Any]:
        if not self.phases:
            raise ContextualExpertAggregationStageError(
                "runtime evidence requires at least one completed phase"
            )
        last = float(self.phases[-1]["elapsed_seconds"])
        return {
            "runtime_evidence_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
            "stage": _stage(stage),
            "clock": "monotonic",
            "limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "strict_boundary": "elapsed_seconds < limit_seconds",
            "phase_elapsed_seconds": copy.deepcopy(self.phases),
            "seconds_before_payload_finalization": last,
            "within_limit_before_payload_finalization": (
                last < RUN_TIME_LIMIT_SECONDS
            ),
            "authoritative_last_check": (
                "seal_exact_bundle immediately before atomic promotion"
            ),
            "network_access": False,
            "news_access": False,
            "llm_calls": 0,
            "api_calls": 0,
            "external_cost_usd": 0.0,
        }


@dataclass(frozen=True)
class StageComputation:
    stage: str
    snapshot: _experiment.LoadedPriceSnapshot
    replays: Mapping[str, _replay.ReplayResult]
    ledgers: Mapping[str, Mapping[str, pd.DataFrame]]
    accounts: Mapping[str, Mapping[str, _ledger.AccountState]]
    episodes: Mapping[str, Mapping[str, _ledger.EpisodeExtraction]]
    xors: Mapping[str, Mapping[str, _ledger.XorExtraction]]
    fixed_parent_proof: Mapping[str, Any]
    source_bundle_provenance: Mapping[str, Any]
    replay_diagnostics: Mapping[str, Any]
    prefix_continuity_proof: Mapping[str, Any] | None
    confirmation_authorization: Mapping[str, Any] | None
    development_parent_manifest_bytes: bytes | None
    development_parent_checksums_bytes: bytes | None


def _policy_target(
    *,
    stage: str,
    policy: str,
    replays: Mapping[str, _replay.ReplayResult],
    fixed_features: pd.DataFrame,
) -> pd.Series:
    selected = _stage(stage)
    if policy in ARM_ORDER:
        arm = ONLINE_FULL_ARM if selected == DEVELOPMENT_STAGE else policy
        return replays[arm].forecast["learner_target_exposure"].copy()
    column_by_policy = {
        "always_long": "fixed_always_long_target_exposure",
        "exact_union_cash": "fixed_union_cash_target_exposure",
        "contextual_only": "fixed_contextual_only_target_exposure",
        "weak_trend_only": "fixed_weak_trend_only_target_exposure",
    }
    if policy == "aapl_buy_hold":
        return pd.Series(1.0, index=fixed_features.index, dtype=float)
    try:
        return fixed_features[column_by_policy[policy]].copy()
    except KeyError as exc:
        raise ContextualExpertAggregationStageError(
            f"unknown frozen policy: {policy}"
        ) from exc


def _fixed_features_for_replays(
    replays: Mapping[str, _replay.ReplayResult]
) -> pd.DataFrame:
    online = replays[ONLINE_FULL_ARM].opportunity_frame
    for arm in ARM_ORDER:
        if not replays[arm].opportunity_frame.equals(online):
            raise ContextualExpertAggregationStageError(
                "stage replay arms changed the action-independent fixed features"
            )
    return online


def _development_replays(
    frame: pd.DataFrame,
) -> tuple[dict[str, _replay.ReplayResult], dict[str, Any]]:
    online = _replay.replay_from_empty(frame)
    replays = {arm: online for arm in ARM_ORDER}
    equality = {
        arm: asdict(_replay.prove_replay_prefix(online, replays[arm]))
        for arm in ARM_ORDER
    }
    if not all(value["passed"] for value in equality.values()):
        raise ContextualExpertAggregationStageError(
            "development arm seed replay identity failed"
        )
    return replays, {
        "development_arm_replay_identity": equality,
        "seed_target_source_policy": ONLINE_FULL_ARM,
    }


def _run_development_ledgers(
    *,
    snapshot: _experiment.LoadedPriceSnapshot,
    replays: Mapping[str, _replay.ReplayResult],
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, _ledger.AccountState]]]:
    fixed = _fixed_features_for_replays(replays)
    opens = fixed["aapl_adj_open"]
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    accounts: dict[str, dict[str, _ledger.AccountState]] = {}
    for cost in COST_ORDER:
        ledgers[cost] = {}
        accounts[cost] = {}
        for policy in POLICY_ORDER:
            run = _ledger.run_continuous_ledger(
                opens,
                _policy_target(
                    stage=DEVELOPMENT_STAGE,
                    policy=policy,
                    replays=replays,
                    fixed_features=fixed,
                ),
                policy_name=policy,
                cost_bps=COST_BPS[cost],
            )
            ledgers[cost][policy] = run.ledger
            accounts[cost][policy] = run.state
    return ledgers, accounts


def _extract_episodes_and_xors(
    ledgers: Mapping[str, Mapping[str, pd.DataFrame]],
) -> tuple[
    dict[str, dict[str, _ledger.EpisodeExtraction]],
    dict[str, dict[str, _ledger.XorExtraction]],
]:
    episodes: dict[str, dict[str, _ledger.EpisodeExtraction]] = {}
    xors: dict[str, dict[str, _ledger.XorExtraction]] = {}
    comparisons = {
        **{name: name for name in FIXED_POLICY_ORDER},
        **ABLATION_POLICY_BY_COMPARISON,
    }
    for cost in COST_ORDER:
        episodes[cost] = {}
        for policy in POLICY_ORDER:
            start = _ledger.AccountState.initial(
                policy_name=policy, cost_bps=COST_BPS[cost]
            )
            episodes[cost][policy] = _ledger.extract_cash_episodes(
                ledgers[cost][policy], start_state=start
            )
        xors[cost] = {}
        primary_start = _ledger.AccountState.initial(
            policy_name=ONLINE_FULL_ARM, cost_bps=COST_BPS[cost]
        )
        for comparison, policy in comparisons.items():
            comparator_start = _ledger.AccountState.initial(
                policy_name=policy, cost_bps=COST_BPS[cost]
            )
            xors[cost][comparison] = _ledger.extract_signed_xor_episodes(
                ledgers[cost][ONLINE_FULL_ARM],
                ledgers[cost][policy],
                primary_start_state=primary_start,
                comparator_start_state=comparator_start,
            )
    return episodes, xors


def classify_terminal_residuals(
    episodes: Mapping[str, Mapping[str, _ledger.EpisodeExtraction]],
    xors: Mapping[str, Mapping[str, _ledger.XorExtraction]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    """Return sorted deterministic fatal residuals.

    A final-close CASH target that never executes is allowed.  An already
    executed open CASH interval and every partial XOR boundary are fatal.
    """

    selected = _stage(stage)
    active_comparisons = set(FIXED_POLICY_ORDER)
    if selected == CONFIRMATION_STAGE:
        active_comparisons.update(ABLATION_POLICY_BY_COMPARISON)
    reasons: list[dict[str, Any]] = []
    for cost in COST_ORDER:
        for policy in POLICY_ORDER:
            unresolved = episodes[cost][policy].unresolved
            for row in unresolved.to_dict(orient="records"):
                status = row.get("status")
                if status == "pending_cash_entry_unexecuted":
                    continue
                if type(status) is not str or not status.startswith("open_cash"):
                    raise ContextualExpertAggregationStageError(
                        "CASH residual has an unknown status"
                    )
                reasons.append(
                    {
                        "reason_code": "executed_terminal_cash_residual",
                        "cost": cost,
                        "policy": policy,
                        "status": status,
                        "entry_fill_date": str(row["entry_fill_date"]),
                        "mark_date": str(row["mark_date"]),
                    }
                )
        for comparison in sorted(active_comparisons):
            for row in xors[cost][comparison].unresolved.to_dict(orient="records"):
                reasons.append(
                    {
                        "reason_code": "partial_xor_boundary",
                        "cost": cost,
                        "comparison": comparison,
                        "status": str(row["status"]),
                        "entry_fill_date": str(row["entry_fill_date"]),
                        "last_fill_date": str(row["last_fill_date"]),
                        "orientation": str(row["orientation"]),
                    }
                )
    return sorted(
        reasons,
        key=lambda value: tuple(str(value.get(name, "")) for name in (
            "reason_code",
            "cost",
            "policy",
            "comparison",
            "entry_fill_date",
            "last_fill_date",
            "mark_date",
        )),
    )


def _integrity_inventory(stage: str) -> frozenset[str]:
    selected = _stage(stage)
    return (
        _evaluation.DEVELOPMENT_INTEGRITY_CHECKS
        if selected == DEVELOPMENT_STAGE
        else _evaluation.CONFIRMATION_INTEGRITY_CHECKS
    )


def _integrity_evidence(
    *, stage: str, fatal_reasons: Sequence[Mapping[str, Any]]
) -> dict[str, bool]:
    result = {name: True for name in _integrity_inventory(stage)}
    if fatal_reasons:
        result["ledger_episode_and_xor_reconciliation_exact"] = False
    return result


def build_fatal_gate_report(
    *,
    stage: str,
    integrity: Mapping[str, bool],
    fatal_reasons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = _stage(stage)
    if not fatal_reasons:
        raise ContextualExpertAggregationStageError(
            "fatal gate report requires at least one terminal residual"
        )
    expected = _integrity_inventory(selected)
    if set(integrity) != set(expected):
        raise ContextualExpertAggregationStageError(
            "fatal integrity evidence has the wrong frozen inventory"
        )
    checks = {
        **{f"integrity.{name}": bool(integrity[name]) for name in sorted(expected)},
        f"fatal.{NO_EXECUTED_TERMINAL_CASH}": not any(
            item.get("reason_code") == "executed_terminal_cash_residual"
            for item in fatal_reasons
        ),
        f"fatal.{NO_PARTIAL_XOR_BOUNDARY}": not any(
            item.get("reason_code") == "partial_xor_boundary"
            for item in fatal_reasons
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "fatal_gate_schema_version": FATAL_GATE_SCHEMA_VERSION,
        "stage": selected,
        "passed": False,
        "passed_count": sum(bool(value) for value in checks.values()),
        "total_count": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "fatal_integrity_rejection": True,
        "normal_economic_gates_evaluated": False,
        "fatal_reasons": [_jsonable(item) for item in fatal_reasons],
    }


def _normal_gate_report(value: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    report = copy.deepcopy(dict(value))
    report.update(
        {
            "stage": _stage(stage),
            "fatal_integrity_rejection": False,
            "normal_economic_gates_evaluated": True,
            "fatal_reasons": [],
        }
    )
    return report


def _reconcile_evidence(
    computation: StageComputation,
) -> None:
    selected = _stage(computation.stage)
    comparisons = set(FIXED_POLICY_ORDER)
    if selected == CONFIRMATION_STAGE:
        comparisons.update(ABLATION_POLICY_BY_COMPARISON)
    for cost in COST_ORDER:
        for policy in POLICY_ORDER:
            _ledger.assert_no_leverage(
                computation.ledgers[cost][policy],
                start_state=_ledger.AccountState.initial(
                    policy_name=policy, cost_bps=COST_BPS[cost]
                ),
            )
            _ledger.reconcile_complete_cash_episodes(
                computation.ledgers[cost][policy],
                computation.ledgers[cost]["aapl_buy_hold"],
                computation.episodes[cost][policy],
                strategy_start_state=_ledger.AccountState.initial(
                    policy_name=policy, cost_bps=COST_BPS[cost]
                ),
                buy_hold_start_state=_ledger.AccountState.initial(
                    policy_name="aapl_buy_hold", cost_bps=COST_BPS[cost]
                ),
            )
        _ledger.assert_always_long_matches_buy_hold(
            computation.ledgers[cost]["always_long"],
            computation.ledgers[cost]["aapl_buy_hold"],
        )
        for comparison in sorted(comparisons):
            policy = (
                comparison
                if comparison in FIXED_POLICY_ORDER
                else ABLATION_POLICY_BY_COMPARISON[comparison]
            )
            _ledger.reconcile_signed_xor(
                computation.ledgers[cost][ONLINE_FULL_ARM],
                computation.ledgers[cost][policy],
                computation.xors[cost][comparison],
                primary_start_state=_ledger.AccountState.initial(
                    policy_name=ONLINE_FULL_ARM, cost_bps=COST_BPS[cost]
                ),
                comparator_start_state=_ledger.AccountState.initial(
                    policy_name=policy, cost_bps=COST_BPS[cost]
                ),
            )
    for policy in POLICY_ORDER:
        _ledger.assert_cross_cost_action_identity(
            {cost: computation.ledgers[cost][policy] for cost in COST_ORDER}
        )


def _evaluation_evidence(
    computation: StageComputation,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    primary: dict[str, Any] = {}
    fixed: dict[str, Any] = {}
    ablations: dict[str, Any] = {}
    for cost in COST_ORDER:
        benchmark = computation.ledgers[cost]["aapl_buy_hold"]
        primary[cost] = {
            "strategy_ledger": computation.ledgers[cost][ONLINE_FULL_ARM],
            "benchmark_ledger": benchmark,
            "complete_episodes": computation.episodes[cost][ONLINE_FULL_ARM].complete,
        }
        fixed[cost] = {}
        for policy in FIXED_POLICY_ORDER:
            fixed[cost][policy] = {
                "strategy_ledger": computation.ledgers[cost][policy],
                "benchmark_ledger": benchmark,
                "complete_episodes": computation.episodes[cost][policy].complete,
                "learner_minus_comparator_xor": computation.xors[cost][policy].complete,
            }
        ablations[cost] = {}
        for comparison, policy in ABLATION_POLICY_BY_COMPARISON.items():
            ablations[cost][comparison] = {
                "strategy_ledger": computation.ledgers[cost][policy],
                "benchmark_ledger": benchmark,
                "complete_episodes": computation.episodes[cost][policy].complete,
                "learner_minus_comparator_xor": computation.xors[cost][comparison].complete,
            }
    return primary, fixed, ablations


def _gate_and_metrics(
    computation: StageComputation,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    reasons = classify_terminal_residuals(
        computation.episodes, computation.xors, stage=computation.stage
    )
    integrity = _integrity_evidence(stage=computation.stage, fatal_reasons=reasons)
    if reasons:
        gate = build_fatal_gate_report(
            stage=computation.stage,
            integrity=integrity,
            fatal_reasons=reasons,
        )
        metrics = {
            "metrics_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
            "stage": computation.stage,
            "status": "not_scored_due_to_fatal_terminal_integrity",
            "normal_economic_gates_evaluated": False,
            "fatal_reasons": reasons,
        }
        return integrity, gate, metrics

    _reconcile_evidence(computation)
    primary, fixed, ablations = _evaluation_evidence(computation)
    if computation.stage == DEVELOPMENT_STAGE:
        raw = _evaluation.apply_development_gates(
            primary,
            fixed_comparator_evidence=fixed,
            integrity=integrity,
        )
    else:
        raw = _evaluation.apply_confirmation_gates(
            primary,
            fixed_comparator_evidence=fixed,
            ablation_evidence=ablations,
            integrity=integrity,
        )
    gate = _normal_gate_report(raw, stage=computation.stage)
    _evaluation.require_report_status(gate, expected_pass=bool(gate["passed"]))
    metrics = {
        "metrics_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "stage": computation.stage,
        "status": "scored",
        "normal_economic_gates_evaluated": True,
        "policy_summaries": copy.deepcopy(gate.get("policy_summaries", {})),
        "fixed_comparator_differences": copy.deepcopy(
            gate.get("fixed_comparator_differences", {})
        ),
        "best_fixed_comparator": copy.deepcopy(
            gate.get("best_fixed_comparator")
        ),
        "ablation_diagnostics": copy.deepcopy(
            gate.get("ablation_diagnostics", {})
        ),
        "learner_minus_union_year_edges": copy.deepcopy(
            gate.get("learner_minus_union_year_edges", {})
        ),
    }
    return integrity, gate, metrics


PARENT_PROJECTION_COLUMNS = (
    *_replay.FIXED_CAUSAL_SIGNAL_COLUMNS,
    *_replay.COMPARATOR_TARGET_COLUMNS,
)
PARENT_PROJECTION_SCHEMA = _artifacts.CanonicalTableSchema(
    columns=PARENT_PROJECTION_COLUMNS,
    column_types=tuple(
        _fixed_feature_types()[_replay.FIXED_FEATURE_COLUMNS.index(name)]
        for name in PARENT_PROJECTION_COLUMNS
    ),
    index_name="decision_date",
    index_type="iso_date",
)


def _source_bundle_for_stage(stage: str) -> _experiment.BundleIdentity:
    return (
        _experiment.DEVELOPMENT_SOURCE_BUNDLE
        if _stage(stage) == DEVELOPMENT_STAGE
        else _experiment.CONFIRMATION_SOURCE_BUNDLE
    )


def _source_bundle_provenance(
    repo_root: Path, *, stage: str
) -> dict[str, Any]:
    selected = _stage(stage)
    identity = _source_bundle_for_stage(selected)
    bundle = _experiment.verify_frozen_bundle_identity(repo_root, identity)
    result: dict[str, Any] = {
        "source_bundle_contract_version": identity.contract_version,
        "source_bundle_stage": identity.stage,
        "source_bundle_manifest_path": identity.manifest_path.as_posix(),
        "source_bundle_manifest_sha256": bundle.manifest["manifest_sha256"],
        "source_bundle_manifest_file_sha256": identity.manifest_file_sha256,
        "source_bundle_checksums_file_sha256": identity.checksums_file_sha256,
        "source_forecast_filename": PARENT_FORECAST_BY_STAGE[selected],
        "source_forecast_sha256": bundle.payload_sha256[
            PARENT_FORECAST_BY_STAGE[selected]
        ],
    }
    if selected == CONFIRMATION_STAGE:
        parent_identity = _experiment.CONFIRMATION_SOURCE_PARENT_BUNDLE
        parent = _experiment.verify_frozen_bundle_identity(repo_root, parent_identity)
        _experiment.require_parent_link(
            bundle,
            parent,
            embedded_parent_filename=(
                _experiment.CONFIRMATION_PRICE_SPEC.embedded_parent_filename
            ),
        )
        result["source_parent_bundle"] = {
            "manifest_path": parent_identity.manifest_path.as_posix(),
            "manifest_sha256": parent.manifest["manifest_sha256"],
            "manifest_file_sha256": parent_identity.manifest_file_sha256,
            "checksums_file_sha256": parent_identity.checksums_file_sha256,
        }
    else:
        result["source_parent_bundle"] = None
    result["provenance_sha256"] = _sha256_json(result)
    return result


def _strict_parent_projection(
    raw: pd.DataFrame, *, expected_index: pd.DatetimeIndex
) -> pd.DataFrame:
    required_parent = {
        "decision_date",
        *_replay.FIXED_CAUSAL_SIGNAL_COLUMNS,
        *PARENT_TARGET_COLUMN_MAP.values(),
    }
    if not required_parent.issubset(raw.columns):
        raise ContextualExpertAggregationStageError(
            "sealed source forecast omits a fixed causal projection column"
        )
    date_values = raw["decision_date"].tolist()
    if any(type(value) is not str for value in date_values):
        raise ContextualExpertAggregationStageError(
            "sealed source forecast dates are not exact strings"
        )
    try:
        index = pd.DatetimeIndex(date_values, name="decision_date")
    except (TypeError, ValueError) as exc:
        raise ContextualExpertAggregationStageError(
            "sealed source forecast dates are invalid"
        ) from exc
    if (
        index.has_duplicates
        or not index.is_monotonic_increasing
        or [value.date().isoformat() for value in index] != date_values
        or not index.equals(expected_index)
    ):
        raise ContextualExpertAggregationStageError(
            "sealed source forecast date sequence changed"
        )
    result = pd.DataFrame(index=index)
    fixed_types = dict(zip(PARENT_PROJECTION_COLUMNS, PARENT_PROJECTION_SCHEMA.column_types))
    for name in _replay.FIXED_CAUSAL_SIGNAL_COLUMNS:
        series = raw[name]
        if fixed_types[name] == "bool":
            if not series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ContextualExpertAggregationStageError(
                    f"sealed source forecast {name} is not exactly boolean"
                )
            result[name] = series.map(bool).to_numpy()
        elif fixed_types[name] == "nullable_float":
            values = pd.to_numeric(series, errors="raise").astype(float)
            if np.isinf(values.to_numpy()).any():
                raise ContextualExpertAggregationStageError(
                    f"sealed source forecast {name} is infinite"
                )
            result[name] = values.to_numpy()
        else:
            values = pd.to_numeric(series, errors="raise").astype(float)
            if not np.isfinite(values.to_numpy()).all():
                raise ContextualExpertAggregationStageError(
                    f"sealed source forecast {name} is nonfinite"
                )
            result[name] = values.to_numpy()
    for generated_name, source_name in PARENT_TARGET_COLUMN_MAP.items():
        values = pd.to_numeric(raw[source_name], errors="raise").astype(float)
        if not values.isin([0.0, 1.0]).all():
            raise ContextualExpertAggregationStageError(
                "sealed parent comparator target is not binary"
            )
        result[generated_name] = values.to_numpy()
    return result.loc[:, list(PARENT_PROJECTION_COLUMNS)]


def _fixed_parent_prefix_proof(
    repo_root: Path,
    *,
    stage: str,
    fixed_features: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = _stage(stage)
    identity = _source_bundle_for_stage(selected)
    source = _experiment.verify_frozen_bundle_identity(repo_root, identity)
    filename = PARENT_FORECAST_BY_STAGE[selected]
    try:
        parent_bytes = (source.directory / filename).read_bytes()
        raw = pd.read_csv(
            io.BytesIO(parent_bytes),
            dtype={"decision_date": str},
            float_precision="round_trip",
        )
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        raise ContextualExpertAggregationStageError(
            "sealed source forecast is unreadable"
        ) from exc
    if _experiment.sha256_bytes(parent_bytes) != source.payload_sha256.get(filename):
        raise ContextualExpertAggregationStageError(
            "sealed source forecast bytes changed after bundle verification"
        )
    generated = fixed_features.loc[:, list(PARENT_PROJECTION_COLUMNS)].copy()
    generated.index = generated.index.rename("decision_date")
    parent = _strict_parent_projection(raw, expected_index=generated.index)
    generated_bytes = table_bytes(generated, schema=PARENT_PROJECTION_SCHEMA)
    parent_projection_bytes = table_bytes(parent, schema=PARENT_PROJECTION_SCHEMA)
    if parent_projection_bytes != generated_bytes:
        raise ContextualExpertAggregationStageError(
            "fixed causal signals or comparator targets differ from the sealed parent"
        )
    proof = {
        "proof_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "stage": selected,
        "source_manifest_sha256": source.manifest["manifest_sha256"],
        "source_forecast_filename": filename,
        "source_forecast_payload_sha256": source.payload_sha256[filename],
        "compared_rows": len(generated),
        "first_decision_date": generated.index[0].date().isoformat(),
        "last_decision_date": generated.index[-1].date().isoformat(),
        "column_order": list(PARENT_PROJECTION_COLUMNS),
        "parent_projection_sha256": _experiment.sha256_bytes(
            parent_projection_bytes
        ),
        "generated_projection_sha256": _experiment.sha256_bytes(generated_bytes),
        "exact": True,
    }
    proof["proof_sha256"] = _sha256_json(proof)
    return proof, _source_bundle_provenance(repo_root, stage=selected)


def _replay_diagnostics_payload(
    *,
    stage: str,
    replays: Mapping[str, _replay.ReplayResult],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "diagnostics_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "stage": _stage(stage),
        "arm_order": list(ARM_ORDER),
        "by_arm": {
            arm: _jsonable(asdict(replays[arm].diagnostics)) for arm in ARM_ORDER
        },
        "extra_proofs": _jsonable(extra),
    }
    payload["diagnostics_sha256"] = _sha256_json(payload)
    return payload


def _compute_development(
    repo_root: Path,
    *,
    trace: _RuntimeTrace,
) -> StageComputation:
    snapshot = _experiment.load_authorized_prices(
        repo_root, stage=DEVELOPMENT_STAGE
    )
    trace.mark("development authorized input loaded")
    replays, replay_extra = _development_replays(snapshot.frame)
    trace.mark("development replay completed")
    fixed = _fixed_features_for_replays(replays)
    parent_proof, source_provenance = _fixed_parent_prefix_proof(
        repo_root,
        stage=DEVELOPMENT_STAGE,
        fixed_features=fixed,
    )
    trace.mark("development fixed parent prefix proved")
    ledgers, accounts = _run_development_ledgers(
        snapshot=snapshot, replays=replays
    )
    episodes, xors = _extract_episodes_and_xors(ledgers)
    trace.mark("development ledgers and differences completed")
    return StageComputation(
        stage=DEVELOPMENT_STAGE,
        snapshot=snapshot,
        replays=replays,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        fixed_parent_proof=parent_proof,
        source_bundle_provenance=source_provenance,
        replay_diagnostics=_replay_diagnostics_payload(
            stage=DEVELOPMENT_STAGE,
            replays=replays,
            extra=replay_extra,
        ),
        prefix_continuity_proof=None,
        confirmation_authorization=None,
        development_parent_manifest_bytes=None,
        development_parent_checksums_bytes=None,
    )


def _read_development_bundle(
    repo_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> _experiment.VerifiedBundle:
    return _experiment.verify_exact_bundle(
        repo_root / OUTPUT_DIRECTORY_BY_STAGE[DEVELOPMENT_STAGE],
        expected_contract_version=CONTRACT_VERSION,
        expected_stage=DEVELOPMENT_STAGE,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_payload_names=_artifacts.DEVELOPMENT_PAYLOAD_NAMES,
        require_stage_pass=True,
        repo_root=repo_root,
    )


def _bundle_payload(bundle: _experiment.VerifiedBundle, filename: str) -> bytes:
    if filename not in bundle.payload_sha256:
        raise ContextualExpertAggregationStageError(
            f"development parent omits required payload: {filename}"
        )
    try:
        payload = (bundle.directory / filename).read_bytes()
    except OSError as exc:
        raise ContextualExpertAggregationStageError(
            f"development parent payload is unreadable: {filename}"
        ) from exc
    if _experiment.sha256_bytes(payload) != bundle.payload_sha256[filename]:
        raise ContextualExpertAggregationStageError(
            f"development parent payload changed after verification: {filename}"
        )
    return payload


def _require_authorized_checkpoint_payload(
    payload: bytes,
    checkpoint: _artifacts.CompositeStageCheckpoint,
    *,
    authorized_payload_sha256: str,
) -> str:
    """Bind authorization to file bytes, not the checkpoint's inner self-hash."""

    if type(payload) is not bytes or not isinstance(
        checkpoint, _artifacts.CompositeStageCheckpoint
    ):
        raise ContextualExpertAggregationStageError(
            "development checkpoint authorization received invalid evidence"
        )
    observed = _experiment.sha256_bytes(payload)
    if observed != authorized_payload_sha256:
        raise ContextualExpertAggregationStageError(
            "development checkpoint payload changed after confirmation authorization"
        )
    if observed == checkpoint.checkpoint_sha256:
        raise ContextualExpertAggregationStageError(
            "checkpoint payload and internal self-hash domains unexpectedly collide"
        )
    return observed


def _confirmation_authorization_payload(
    repo_root: Path,
    authorization: _experiment.ConfirmationAuthorization,
) -> dict[str, Any]:
    try:
        manifest_relative = authorization.development_manifest_path.relative_to(
            repo_root
        ).as_posix()
        lock_bytes = authorization.attempt_lock_path.read_bytes()
        lock_value = json.loads(lock_bytes.decode("utf-8"))
    except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationStageError(
            "confirmation authorization evidence is unreadable"
        ) from exc
    if _experiment.sha256_bytes(lock_bytes) != authorization.attempt_lock_sha256:
        raise ContextualExpertAggregationStageError(
            "confirmation attempt lock changed after authorization"
        )
    payload = {
        "authorization_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "development_manifest_path": manifest_relative,
        "development_manifest_sha256": authorization.development_manifest_sha256,
        "development_checkpoint_sha256": authorization.development_checkpoint_sha256,
        "development_gate_report_sha256": authorization.development_gate_report_sha256,
        "attempt_identity_sha256": authorization.attempt_identity_sha256,
        "attempt_lock_filename": authorization.attempt_lock_path.name,
        "attempt_lock_sha256": authorization.attempt_lock_sha256,
        "attempt_lock_parent_fsync_supported": (
            authorization.attempt_lock_parent_fsync_supported
        ),
        "prelock_git_identity": dict(authorization.prelock_git_identity),
        "postlock_git_identity": dict(authorization.git_identity),
        "development_verification": dict(authorization.development_verification),
        "attempt_lock_payload": lock_value,
        "confirmation_input_opened_before_lock": False,
    }
    payload["authorization_sha256"] = _sha256_json(payload)
    return payload


def _require_development_replay_artifacts(
    bundle: _experiment.VerifiedBundle,
    *,
    regenerated: _replay.ReplayResult,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    expected_forecast = _decision_table_bytes(
        regenerated.forecast, schema=FORECAST_SCHEMA
    )
    expected_matured = table_bytes(
        regenerated.matured_lessons, schema=MATURED_EVENT_SCHEMA
    )
    expected_fixed = _decision_table_bytes(
        regenerated.opportunity_frame, schema=FIXED_FEATURE_SCHEMA
    )
    fixed_payload = _bundle_payload(
        bundle, "development_fixed_features.table.json"
    )
    if fixed_payload != expected_fixed:
        raise ContextualExpertAggregationStageError(
            "development parent fixed features do not replay"
        )
    checks["fixed_features_sha256"] = _experiment.sha256_bytes(expected_fixed)
    expected_comparators = _decision_table_bytes(
        regenerated.opportunity_frame.loc[
            :, list(_replay.COMPARATOR_TARGET_COLUMNS)
        ],
        schema=FIXED_COMPARATOR_SCHEMA,
    )
    comparator_payload = _bundle_payload(
        bundle, "development_forecast__fixed_comparators.table.json"
    )
    if comparator_payload != expected_comparators:
        raise ContextualExpertAggregationStageError(
            "development parent fixed comparator forecast does not replay"
        )
    checks["fixed_comparators_sha256"] = _experiment.sha256_bytes(
        comparator_payload
    )
    for arm in ARM_ORDER:
        payload = _bundle_payload(
            bundle, f"development_forecast__{arm}.table.json"
        )
        if payload != expected_forecast:
            raise ContextualExpertAggregationStageError(
                f"development parent forecast does not replay for {arm}"
            )
        checks[f"forecast__{arm}_sha256"] = _experiment.sha256_bytes(payload)
        matured_payload = _bundle_payload(
            bundle, f"development_matured_lessons__{arm}.table.json"
        )
        if matured_payload != expected_matured:
            raise ContextualExpertAggregationStageError(
                f"development parent matured lessons do not replay for {arm}"
            )
        checks[f"matured_lessons__{arm}_sha256"] = (
            _experiment.sha256_bytes(matured_payload)
        )
    return checks


def _verified_parent_metadata_bytes(
    bundle: _experiment.VerifiedBundle,
) -> tuple[bytes, bytes]:
    try:
        manifest_bytes = bundle.manifest_path.read_bytes()
        checksums_bytes = (bundle.directory / "checksums.json").read_bytes()
        manifest_value = json.loads(manifest_bytes.decode("utf-8"))
        checksums_value = json.loads(checksums_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationStageError(
            "verified development parent metadata became unreadable"
        ) from exc
    if (
        _experiment.sha256_bytes(manifest_bytes)
        != bundle.checksums["stage_manifest.json"]
        or manifest_value != dict(bundle.manifest)
        or checksums_value != dict(bundle.checksums)
    ):
        raise ContextualExpertAggregationStageError(
            "verified development parent metadata changed after verification"
        )
    return manifest_bytes, checksums_bytes


def _run_confirmation_ledgers(
    *,
    development_bundle: _experiment.VerifiedBundle,
    development_snapshot: _experiment.LoadedPriceSnapshot,
    confirmation_snapshot: _experiment.LoadedPriceSnapshot,
    development_replays: Mapping[str, _replay.ReplayResult],
    confirmation_replays: Mapping[str, _replay.ReplayResult],
    development_checkpoint: _artifacts.CompositeStageCheckpoint,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, _ledger.AccountState]]]:
    development_fixed = _fixed_features_for_replays(development_replays)
    confirmation_fixed = _fixed_features_for_replays(confirmation_replays)
    prior_opens = development_fixed["aapl_adj_open"]
    suffix_opens = confirmation_fixed["aapl_adj_open"]
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    accounts: dict[str, dict[str, _ledger.AccountState]] = {}
    for cost in COST_ORDER:
        ledgers[cost] = {}
        accounts[cost] = {}
        for policy in POLICY_ORDER:
            prior_ledger = parse_ledger_table_bytes(
                _bundle_payload(
                    development_bundle,
                    f"development_ledger__{cost}__{policy}.table.json",
                )
            )
            prior_target = _policy_target(
                stage=DEVELOPMENT_STAGE,
                policy=policy,
                replays=development_replays,
                fixed_features=development_fixed,
            )
            suffix_target = _policy_target(
                stage=CONFIRMATION_STAGE,
                policy=policy,
                replays=confirmation_replays,
                fixed_features=confirmation_fixed,
            )
            suffix = _ledger.run_verified_continuation(
                prior_opens,
                prior_target,
                prior_ledger,
                development_checkpoint.administrative_accounts[cost][
                    policy
                ].to_checkpoint(),
                suffix_opens,
                suffix_target,
                policy_name=policy,
                cost_bps=COST_BPS[cost],
            )
            full = pd.concat([prior_ledger, suffix.ledger], ignore_index=True)
            _ledger.verify_ledger(
                full,
                start_state=_ledger.AccountState.initial(
                    policy_name=policy, cost_bps=COST_BPS[cost]
                ),
                expected_end_state=suffix.state,
            )
            ledgers[cost][policy] = full
            accounts[cost][policy] = suffix.state
    return ledgers, accounts


def _compute_confirmation(
    repo_root: Path,
    *,
    trace: _RuntimeTrace,
) -> StageComputation:
    authorization = _experiment.authorize_confirmation_attempt(
        repo_root=repo_root,
        development_manifest_path=repo_root / DEVELOPMENT_MANIFEST_PATH,
    )
    trace.mark("confirmation attempt durably authorized")
    development_bundle = _read_development_bundle(
        repo_root,
        expected_manifest_sha256=authorization.development_manifest_sha256,
    )
    development_snapshot = _experiment.load_authorized_prices(
        repo_root, stage=DEVELOPMENT_STAGE
    )
    development_online = _replay.replay_from_empty(development_snapshot.frame)
    development_replays = {arm: development_online for arm in ARM_ORDER}
    checkpoint_payload = _bundle_payload(
        development_bundle, "development_checkpoint_through_2018.json"
    )
    development_checkpoint = _artifacts.parse_composite_stage_checkpoint_bytes(
        checkpoint_payload
    )
    checkpoint_payload_sha256 = _require_authorized_checkpoint_payload(
        checkpoint_payload,
        development_checkpoint,
        authorized_payload_sha256=authorization.development_checkpoint_sha256,
    )
    replay_proofs: dict[str, Any] = {}
    for arm in ARM_ORDER:
        sealed_checkpoint = development_checkpoint.replay_checkpoints[arm]
        if sealed_checkpoint.to_dict() != development_online.checkpoint.to_dict():
            raise ContextualExpertAggregationStageError(
                f"development checkpoint does not replay for confirmation arm {arm}"
            )
        proof = _replay.prove_replay_prefix(development_online, development_online)
        proof.require()
        replay_proofs[arm] = asdict(proof)
    parent_replay_artifacts = _require_development_replay_artifacts(
        development_bundle, regenerated=development_online
    )
    trace.mark("development checkpoint and account prefix replayed")

    # This is the first call authorized to return any 2019-2023 market value.
    confirmation_snapshot = _experiment.load_authorized_prices(
        repo_root,
        stage=CONFIRMATION_STAGE,
        development_snapshot=development_snapshot,
        confirmation_authorization=authorization,
    )
    suffix_frame = confirmation_snapshot.frame.loc[
        confirmation_snapshot.frame.index
        > pd.Timestamp(development_online.checkpoint.checkpoint_date)
    ].copy()
    if suffix_frame.empty:
        raise ContextualExpertAggregationStageError(
            "confirmation suffix is unexpectedly empty"
        )
    forks = _replay.fork_confirmation_arms(
        suffix_frame,
        development_online.checkpoint,
        historical_prefix=development_snapshot.frame,
    )
    confirmation_replays = dict(forks.arms)
    full_online = _replay.replay_from_empty(confirmation_snapshot.frame)
    resume = _replay.verify_resume_equivalence(
        full_online, confirmation_replays[ONLINE_FULL_ARM]
    )
    resume.require()
    trace.mark("confirmation replay forks completed")
    fixed = _fixed_features_for_replays(confirmation_replays)
    parent_proof, source_provenance = _fixed_parent_prefix_proof(
        repo_root,
        stage=CONFIRMATION_STAGE,
        fixed_features=fixed,
    )
    ledgers, accounts = _run_confirmation_ledgers(
        development_bundle=development_bundle,
        development_snapshot=development_snapshot,
        confirmation_snapshot=confirmation_snapshot,
        development_replays=development_replays,
        confirmation_replays=confirmation_replays,
        development_checkpoint=development_checkpoint,
    )
    episodes, xors = _extract_episodes_and_xors(ledgers)
    trace.mark("confirmation ledgers and differences completed")
    prefix_proof = {
        "proof_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "development_manifest_sha256": development_bundle.manifest[
            "manifest_sha256"
        ],
        "development_checkpoint_payload_sha256": checkpoint_payload_sha256,
        "development_checkpoint_self_sha256": (
            development_checkpoint.checkpoint_sha256
        ),
        "fork_parent_checkpoint_sha256": forks.checkpoint_sha256,
        "arm_replay_prefix_proofs": replay_proofs,
        "online_resume_equivalence": asdict(resume),
        "development_parent_replay_artifacts": parent_replay_artifacts,
        "development_account_prefix_verified_for_every_cost_policy": True,
        "confirmation_suffix_first_session": suffix_frame.index[0].date().isoformat(),
        "confirmation_suffix_last_session": suffix_frame.index[-1].date().isoformat(),
    }
    prefix_proof["proof_sha256"] = _sha256_json(prefix_proof)
    auth_payload = _confirmation_authorization_payload(repo_root, authorization)
    parent_manifest_bytes, parent_checksums_bytes = (
        _verified_parent_metadata_bytes(development_bundle)
    )
    return StageComputation(
        stage=CONFIRMATION_STAGE,
        snapshot=confirmation_snapshot,
        replays=confirmation_replays,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        fixed_parent_proof=parent_proof,
        source_bundle_provenance=source_provenance,
        replay_diagnostics=_replay_diagnostics_payload(
            stage=CONFIRMATION_STAGE,
            replays=confirmation_replays,
            extra={
                "common_fork_checkpoint_sha256": forks.checkpoint_sha256,
                "online_resume_equivalence": asdict(resume),
                "frozen_post_cutoff_admitted_events": int(
                    confirmation_replays[FROZEN_2018_ARM].diagnostics.admitted_events
                ),
            },
        ),
        prefix_continuity_proof=prefix_proof,
        confirmation_authorization=auth_payload,
        development_parent_manifest_bytes=parent_manifest_bytes,
        development_parent_checksums_bytes=parent_checksums_bytes,
    )


def _diagnostic_frame(
    replays: Mapping[str, _replay.ReplayResult]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    selected = [name for name in DIAGNOSTIC_COLUMNS if name not in {"arm", "decision_date"}]
    for arm in ARM_ORDER:
        value = replays[arm].forecast.loc[:, selected].copy()
        value.insert(0, "decision_date", value.index.map(lambda item: item.date().isoformat()))
        value.insert(0, "arm", arm)
        value = value.reset_index(drop=True)
        frames.append(value.loc[:, list(DIAGNOSTIC_COLUMNS)])
    return pd.concat(frames, ignore_index=True)


def _episode_payload(
    computation: StageComputation, *, cost: str
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy in POLICY_ORDER:
        extraction = computation.episodes[cost][policy]
        policies[policy] = {
            "complete_columns": list(_ledger.EPISODE_COLUMNS),
            "complete": _jsonable(extraction.complete.to_dict(orient="records")),
            "unresolved_columns": list(_ledger.UNRESOLVED_EPISODE_COLUMNS),
            "unresolved": _jsonable(extraction.unresolved.to_dict(orient="records")),
        }
    payload = {
        "episode_artifact_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "stage": computation.stage,
        "cost": cost,
        "cost_bps": COST_BPS[cost],
        "policy_order": list(POLICY_ORDER),
        "policies": policies,
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    return payload


def _active_comparison_order(stage: str) -> tuple[str, ...]:
    return (
        tuple(FIXED_POLICY_ORDER)
        if _stage(stage) == DEVELOPMENT_STAGE
        else (
            *tuple(FIXED_POLICY_ORDER),
            *tuple(_evaluation.ABLATION_COMPARISON_NAMES),
        )
    )


def _xor_payload(
    computation: StageComputation, *, cost: str
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for comparison in _active_comparison_order(computation.stage):
        extraction = computation.xors[cost][comparison]
        comparisons[comparison] = {
            "complete_columns": list(_ledger.XOR_COLUMNS),
            "complete": _jsonable(extraction.complete.to_dict(orient="records")),
            "unresolved_columns": list(_ledger.UNRESOLVED_XOR_COLUMNS),
            "unresolved": _jsonable(extraction.unresolved.to_dict(orient="records")),
        }
    payload = {
        "xor_artifact_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "stage": computation.stage,
        "cost": cost,
        "cost_bps": COST_BPS[cost],
        "comparison_order": list(_active_comparison_order(computation.stage)),
        "comparisons": comparisons,
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    return payload


def _pending_lessons_payload(computation: StageComputation) -> dict[str, Any]:
    payload = {
        "pending_artifact_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "stage": computation.stage,
        "arm_order": list(ARM_ORDER),
        "by_arm": {
            arm: _jsonable(list(computation.replays[arm].checkpoint.pending_lessons))
            for arm in ARM_ORDER
        },
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    return payload


def _integrity_payload(
    *,
    stage: str,
    integrity: Mapping[str, bool],
    computation: StageComputation,
) -> dict[str, Any]:
    payload = {
        "integrity_evidence_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "stage": _stage(stage),
        "checks": {name: bool(integrity[name]) for name in sorted(integrity)},
        "fixed_parent_prefix_proof_sha256": computation.fixed_parent_proof[
            "proof_sha256"
        ],
        "replay_diagnostics_sha256": computation.replay_diagnostics[
            "diagnostics_sha256"
        ],
        "prefix_continuity_proof_sha256": (
            None
            if computation.prefix_continuity_proof is None
            else computation.prefix_continuity_proof["proof_sha256"]
        ),
        "zero_use": {
            "network_access": False,
            "news_access": False,
            "llm_calls": 0,
            "api_calls": 0,
            "external_cost_usd": 0.0,
        },
    }
    payload["integrity_evidence_sha256"] = _sha256_json(payload)
    return payload


def _base_payloads(
    computation: StageComputation,
    *,
    integrity: Mapping[str, bool],
    gate_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    stage = computation.stage
    token = _artifacts.CUTOFF_TOKEN_BY_STAGE[stage]
    fixed = _fixed_features_for_replays(computation.replays)
    checkpoint = _artifacts.build_composite_stage_checkpoint(
        stage=stage,
        replay_checkpoints={
            arm: computation.replays[arm].checkpoint for arm in ARM_ORDER
        },
        administrative_accounts=computation.accounts,
    )
    checkpoint_bytes = _artifacts.composite_stage_checkpoint_bytes(checkpoint)
    payloads: dict[str, bytes] = {
        ".gitattributes": _artifacts.GIT_ATTRIBUTES_BYTES,
        "input_provenance.json": _json_bytes(dict(computation.snapshot.provenance)),
        "source_bundle_provenance.json": _json_bytes(
            computation.source_bundle_provenance
        ),
        f"{stage}_prices_through_{token}.csv": computation.snapshot.canonical_csv_bytes,
        f"{stage}_fixed_features.table.json": _decision_table_bytes(
            fixed, schema=FIXED_FEATURE_SCHEMA
        ),
        f"{stage}_fixed_parent_prefix_proof.json": _json_bytes(
            computation.fixed_parent_proof
        ),
        f"{stage}_forecast__fixed_comparators.table.json": _decision_table_bytes(
            fixed.loc[:, list(_replay.COMPARATOR_TARGET_COLUMNS)],
            schema=FIXED_COMPARATOR_SCHEMA,
        ),
        f"{stage}_state_weight_diagnostics.table.json": table_bytes(
            _diagnostic_frame(computation.replays), schema=DIAGNOSTIC_SCHEMA
        ),
        f"{stage}_replay_diagnostics.json": _json_bytes(
            computation.replay_diagnostics
        ),
        f"{stage}_pending_lessons.json": _json_bytes(
            _pending_lessons_payload(computation)
        ),
        f"{stage}_checkpoint_through_{token}.json": checkpoint_bytes,
        f"{stage}_metrics.json": _json_bytes(metrics),
        f"{stage}_integrity_evidence.json": _json_bytes(
            _integrity_payload(
                stage=stage,
                integrity=integrity,
                computation=computation,
            )
        ),
        f"{stage}_gate_report.json": _json_bytes(gate_report),
    }
    for arm in ARM_ORDER:
        payloads[f"{stage}_forecast__{arm}.table.json"] = _decision_table_bytes(
            computation.replays[arm].forecast, schema=FORECAST_SCHEMA
        )
        payloads[f"{stage}_matured_lessons__{arm}.table.json"] = table_bytes(
            computation.replays[arm].matured_lessons,
            schema=MATURED_EVENT_SCHEMA,
        )
    for cost in COST_ORDER:
        for policy in POLICY_ORDER:
            payloads[f"{stage}_ledger__{cost}__{policy}.table.json"] = (
                ledger_table_bytes(computation.ledgers[cost][policy])
            )
        payloads[f"{stage}_episodes__{cost}.json"] = _json_bytes(
            _episode_payload(computation, cost=cost)
        )
        payloads[f"{stage}_xor__{cost}.json"] = _json_bytes(
            _xor_payload(computation, cost=cost)
        )
    if stage == DEVELOPMENT_STAGE:
        seed = checkpoint["arm_seed_equivalence"]
        if not isinstance(seed, Mapping):
            raise ContextualExpertAggregationStageError(
                "development checkpoint omits arm seed equivalence"
            )
        payloads["development_arm_seed_equivalence.json"] = _json_bytes(seed)
    else:
        if (
            computation.development_parent_manifest_bytes is None
            or computation.development_parent_checksums_bytes is None
            or computation.confirmation_authorization is None
            or computation.prefix_continuity_proof is None
        ):
            raise ContextualExpertAggregationStageError(
                "confirmation computation omits parent or authorization evidence"
            )
        payloads["development_parent_manifest.json"] = (
            computation.development_parent_manifest_bytes
        )
        payloads["development_parent_checksums.json"] = (
            computation.development_parent_checksums_bytes
        )
        payloads["confirmation_attempt_authorization.json"] = _json_bytes(
            computation.confirmation_authorization
        )
        payloads["confirmation_prefix_continuity_proof.json"] = _json_bytes(
            computation.prefix_continuity_proof
        )
    return payloads, checkpoint


def _report(
    *,
    computation: StageComputation,
    gate_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    passed = bool(gate_report["passed"])
    report = {
        "report_schema_version": STAGE_EVIDENCE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "stage": computation.stage,
        "run_id": RUN_ID_BY_STAGE[computation.stage],
        "status": "PASSED" if passed else "REJECTED",
        "stage_pass": passed,
        "evidence_classification": (
            "through_2018_training_diagnostic"
            if computation.stage == DEVELOPMENT_STAGE
            else "one_shot_untouched_2019_2023_confirmation"
        ),
        "calendar_cutoff": _artifacts.CHECKPOINT_CUTOFF_BY_STAGE[
            computation.stage
        ],
        "physical_data_end": _artifacts.LAST_OBSERVED_SESSION_BY_STAGE[
            computation.stage
        ],
        "post_stage_market_values_accessed": False,
        "continuous_account_start": _ledger.ACCOUNT_INCEPTION_DATE,
        "account_reset_count_after_inception": 0,
        "gate_report": _jsonable(gate_report),
        "metrics": _jsonable(metrics),
        "runtime_cost_evidence": _jsonable(runtime),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "source_bundle_provenance_sha256": computation.source_bundle_provenance[
            "provenance_sha256"
        ],
        "historical_results_authorize_real_capital": False,
    }
    report["report_sha256"] = _sha256_json(report)
    return report


def _manifest_fields(
    *,
    computation: StageComputation,
    git_identity: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    stage = computation.stage
    return {
        "stage": stage,
        "stage_pass": bool(gate_report["passed"]),
        "run_id": RUN_ID_BY_STAGE[stage],
        "frozen_command": list(FROZEN_COMMAND_BY_STAGE[stage]),
        "evidence_classification": (
            "through_2018_training_diagnostic"
            if stage == DEVELOPMENT_STAGE
            else "one_shot_untouched_2019_2023_confirmation"
        ),
        "git_identity": dict(git_identity),
        "source_provenance": dict(computation.snapshot.provenance),
        "source_bundle_provenance": dict(computation.source_bundle_provenance),
        "parent_manifest_sha256": (
            None
            if stage == DEVELOPMENT_STAGE
            else computation.confirmation_authorization[
                "development_manifest_sha256"
            ]
        ),
        "bounded_result_sha256": computation.snapshot.spec.canonical_sha256,
        "input_raw_sha256": computation.snapshot.raw_sha256,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "gate_report_sha256": _sha256_json(gate_report),
        "runtime_cost_evidence_sha256": _sha256_json(runtime),
        "model_constants": dict(MODEL_CONSTANTS),
        "arm_order": list(ARM_ORDER),
        "fixed_policy_order": list(FIXED_POLICY_ORDER),
        "policy_order": list(POLICY_ORDER),
        "cost_order": list(COST_ORDER),
        "cost_bps": dict(COST_BPS),
        "execution": {
            "asset": "AAPL",
            "actions": ["LONG_100_PERCENT", "CASH_100_PERCENT"],
            "maximum_target_exposure": 1.0,
            "shorting": False,
            "leverage": False,
            "borrowing": False,
            "negative_cash": False,
            "forced_terminal_sale": False,
            "network_access": False,
            "news_access": False,
            "llm_calls": 0,
            "api_calls": 0,
            "external_cost_usd": 0.0,
            "runtime_limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "promotion_requires_strict_pre_rename_deadline": True,
        },
    }


def _validate_table_payload(filename: str, payload: bytes) -> None:
    if "_ledger__" in filename:
        parse_ledger_table_bytes(payload)
    elif "_forecast__fixed_comparators" in filename:
        _artifacts.parse_canonical_table_bytes(
            payload, schema=FIXED_COMPARATOR_SCHEMA
        )
    elif "_forecast__" in filename:
        _artifacts.parse_canonical_table_bytes(payload, schema=FORECAST_SCHEMA)
    elif "_fixed_features" in filename:
        _artifacts.parse_canonical_table_bytes(
            payload, schema=FIXED_FEATURE_SCHEMA
        )
    elif "_matured_lessons__" in filename:
        _artifacts.parse_canonical_table_bytes(
            payload, schema=MATURED_EVENT_SCHEMA
        )
    elif "_state_weight_diagnostics" in filename:
        _artifacts.parse_canonical_table_bytes(payload, schema=DIAGNOSTIC_SCHEMA)
    else:
        raise ContextualExpertAggregationStageError(
            f"unknown canonical table payload: {filename}"
        )


def validate_private_stage_bundle(
    directory: Path,
    *,
    stage: str,
    expected_payloads: Mapping[str, bytes],
) -> _experiment.VerifiedBundle:
    """Read back a private bundle and validate every exact payload representation."""

    selected = _stage(stage)
    expected_names = _artifacts.payload_names_for_stage(selected)
    if set(expected_payloads) != set(expected_names):
        raise ContextualExpertAggregationStageError(
            "private semantic hook received the wrong payload inventory"
        )
    verified = _experiment.verify_exact_bundle(
        directory,
        expected_contract_version=CONTRACT_VERSION,
        expected_stage=selected,
        expected_payload_names=expected_names,
    )
    for filename in sorted(expected_names):
        try:
            payload = (directory / filename).read_bytes()
        except OSError as exc:
            raise ContextualExpertAggregationStageError(
                f"private bundle payload is unreadable: {filename}"
            ) from exc
        if payload != expected_payloads[filename]:
            raise ContextualExpertAggregationStageError(
                f"private bundle payload changed before promotion: {filename}"
            )
        if filename.endswith(".table.json"):
            _validate_table_payload(filename, payload)
        elif filename.endswith("_checkpoint_through_2018.json") or filename.endswith(
            "_checkpoint_through_2023.json"
        ):
            _artifacts.parse_composite_stage_checkpoint_bytes(payload)
        elif filename in {
            ".gitattributes",
            f"{selected}_prices_through_{_artifacts.CUTOFF_TOKEN_BY_STAGE[selected]}.csv",
        }:
            continue
        elif filename in {
            "development_parent_manifest.json",
            "development_parent_checksums.json",
        }:
            try:
                json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContextualExpertAggregationStageError(
                    f"embedded parent JSON is unreadable: {filename}"
                ) from exc
        else:
            _strict_json(payload, field=filename)
    return verified


def _ensure_output_parent(repo_root: Path) -> Path:
    parent = repo_root / OUTPUT_PARENT
    if parent.exists():
        if parent.is_symlink() or not parent.is_dir():
            raise ContextualExpertAggregationStageError(
                "frozen output parent is not an ordinary directory"
            )
    else:
        if not (repo_root / "e").is_dir():
            raise ContextualExpertAggregationStageError(
                "repository evidence parent is missing"
            )
        parent.mkdir()
    return parent


def run_stage(
    stage: str,
    *,
    repo_root: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one source-frozen stage.  Test hooks are not exposed by the CLI."""

    selected = _stage(stage)
    trace = _RuntimeTrace.start(clock)
    root = Path.cwd().resolve() if repo_root is None else Path(repo_root).resolve()
    if selected == DEVELOPMENT_STAGE:
        git_identity = _experiment.clean_git_identity(root)
        trace.mark("development git authorization")
        _ensure_output_parent(root)
        computation = _compute_development(root, trace=trace)
    else:
        if not (root / OUTPUT_PARENT).is_dir():
            raise ContextualExpertAggregationStageError(
                "confirmation requires the committed development output parent"
            )
        computation = _compute_confirmation(root, trace=trace)
        assert computation.confirmation_authorization is not None
        git_identity = computation.confirmation_authorization[
            "postlock_git_identity"
        ]
    integrity, gate_report, metrics = _gate_and_metrics(computation)
    trace.mark(f"{selected} gates completed")
    payloads, checkpoint = _base_payloads(
        computation,
        integrity=integrity,
        gate_report=gate_report,
        metrics=metrics,
    )
    trace.mark(f"{selected} payload construction completed")
    runtime = trace.evidence(stage=selected)
    payloads[f"{selected}_runtime_cost_evidence.json"] = _json_bytes(runtime)
    report = _report(
        computation=computation,
        gate_report=gate_report,
        metrics=metrics,
        runtime=runtime,
        checkpoint=checkpoint,
    )
    payloads["report.json"] = _json_bytes(report)
    expected_names = _artifacts.payload_names_for_stage(selected)
    if set(payloads) != set(expected_names):
        missing = sorted(set(expected_names) - set(payloads))
        extra = sorted(set(payloads) - set(expected_names))
        raise ContextualExpertAggregationStageError(
            f"stage payload inventory mismatch; missing={missing}, extra={extra}"
        )
    final = root / OUTPUT_DIRECTORY_BY_STAGE[selected]
    temporary = final.parent / f".{final.name}.sealing" / final.name
    sealed = _experiment.seal_exact_bundle(
        final,
        manifest_fields=_manifest_fields(
            computation=computation,
            git_identity=git_identity,
            gate_report=gate_report,
            runtime=runtime,
            checkpoint=checkpoint,
        ),
        payloads=payloads,
        expected_payload_names=expected_names,
        deadline=trace.deadline,
        before_promote=lambda: validate_private_stage_bundle(
            temporary,
            stage=selected,
            expected_payloads=payloads,
        ),
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "stage": selected,
        "stage_pass": bool(gate_report["passed"]),
        "status": report["status"],
        "run_id": RUN_ID_BY_STAGE[selected],
        "artifact_directory": str(sealed.directory),
        "manifest_sha256": sealed.manifest["manifest_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "failed_checks": list(gate_report["failed_checks"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGE_ORDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_stage(args.stage)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["stage_pass"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ABLATION_POLICY_BY_COMPARISON",
    "CONFIRMATION_STAGE",
    "ContextualExpertAggregationStageError",
    "DEVELOPMENT_STAGE",
    "DIAGNOSTIC_SCHEMA",
    "FIXED_COMPARATOR_SCHEMA",
    "FIXED_FEATURE_SCHEMA",
    "FROZEN_COMMAND_BY_STAGE",
    "FORECAST_SCHEMA",
    "LEDGER_SCHEMA",
    "MATURED_EVENT_SCHEMA",
    "MODEL_CONSTANTS",
    "PARENT_PROJECTION_SCHEMA",
    "StageComputation",
    "build_fatal_gate_report",
    "classify_terminal_residuals",
    "ledger_table_bytes",
    "main",
    "parse_ledger_table_bytes",
    "run_stage",
    "table_bytes",
    "validate_private_stage_bundle",
]
