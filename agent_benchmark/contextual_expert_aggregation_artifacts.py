"""Frozen deterministic artifact contract for contextual expert aggregation.

This module owns artifact names, canonical heterogeneous-table bytes, and the
composite model/account checkpoint envelope.  It performs no acquisition,
model stage execution, scoring, sealing, or network access.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contextual_expert_aggregation import (
    CAUSAL_ONLINE_MODE,
    FROZEN_CUTOFF_MODE,
    FULL_MODE,
    GLOBAL_ONLY_MODE,
    LIFETIME_ONLY_MODE,
)
from .contextual_expert_aggregation_ledger import (
    ACCOUNT_INCEPTION_FILL_DATE,
    AccountState,
)
from .contextual_expert_aggregation_replay import (
    CONFIRMATION_ARM_ORDER,
    FROZEN_2018_ARM,
    GLOBAL_ONLY_ARM,
    LIFETIME_ONLY_ARM,
    ONLINE_FULL_ARM,
    ReplayCheckpoint,
)


CONTRACT_VERSION = "aapl-causal-contextual-expert-aggregation-v1"
GIT_ATTRIBUTES_BYTES = b"* -text\n"
TABLE_SCHEMA_VERSION = 1
COMPOSITE_CHECKPOINT_SCHEMA_VERSION = 1
SEED_EQUIVALENCE_SCHEMA_VERSION = 1

DEVELOPMENT_STAGE = "development"
CONFIRMATION_STAGE = "confirmation"
STAGE_ORDER = (DEVELOPMENT_STAGE, CONFIRMATION_STAGE)

ARM_ORDER = tuple(CONFIRMATION_ARM_ORDER)
FIXED_POLICY_ORDER = (
    "always_long",
    "exact_union_cash",
    "contextual_only",
    "weak_trend_only",
)
POLICY_ORDER = (*ARM_ORDER, *FIXED_POLICY_ORDER, "aapl_buy_hold")
COST_ORDER = ("base_5bps", "stress_10bps")
COST_BPS: Mapping[str, float] = MappingProxyType(
    {
        "base_5bps": 5.0,
        "stress_10bps": 10.0,
    }
)

RUN_ID_BY_STAGE: Mapping[str, str] = MappingProxyType(
    {
        DEVELOPMENT_STAGE: "contextual-expert-aggregation-development-v1",
        CONFIRMATION_STAGE: "contextual-expert-aggregation-confirmation-v1",
    }
)
OUTPUT_PARENT = Path("e/aapl_causal_contextual_expert_aggregation_v1")
OUTPUT_DIRECTORY_BY_STAGE: Mapping[str, Path] = MappingProxyType(
    {stage: OUTPUT_PARENT / run_id for stage, run_id in RUN_ID_BY_STAGE.items()}
)
CHECKPOINT_CUTOFF_BY_STAGE: Mapping[str, str] = MappingProxyType(
    {
        DEVELOPMENT_STAGE: "2018-12-31",
        CONFIRMATION_STAGE: "2023-12-31",
    }
)
LAST_OBSERVED_SESSION_BY_STAGE: Mapping[str, str] = MappingProxyType(
    {
        DEVELOPMENT_STAGE: "2018-12-31",
        CONFIRMATION_STAGE: "2023-12-29",
    }
)
SOURCE_START_SESSION_BY_STAGE: Mapping[str, str] = MappingProxyType(
    {
        DEVELOPMENT_STAGE: "1999-03-10",
        CONFIRMATION_STAGE: "1999-03-10",
    }
)
SOURCE_SESSION_COUNT_BY_STAGE: Mapping[str, int] = MappingProxyType(
    {
        DEVELOPMENT_STAGE: 4_986,
        CONFIRMATION_STAGE: 6_244,
    }
)
FIRST_CONFIRMATION_SESSION = "2019-01-02"
CUTOFF_TOKEN_BY_STAGE: Mapping[str, str] = MappingProxyType(
    {
        DEVELOPMENT_STAGE: "2018",
        CONFIRMATION_STAGE: "2023",
    }
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

SCALAR_TYPES = (
    "bool",
    "int",
    "float",
    "string",
    "iso_date",
    "nullable_bool",
    "nullable_int",
    "nullable_float",
    "nullable_string",
    "nullable_iso_date",
)
_NON_NULL_TYPES = frozenset(
    {"bool", "int", "float", "string", "iso_date"}
)
_TABLE_ENVELOPE_FIELDS = {
    "table_schema_version",
    "index_name",
    "index_type",
    "columns",
    "column_types",
    "index_values",
    "rows",
}

_ACCOUNT_IDENTITY_FIELDS = frozenset({"policy_name", "ledger_tip_sha256"})
ECONOMIC_ACCOUNT_FIELDS = tuple(
    name
    for name in AccountState.__dataclass_fields__
    if name not in _ACCOUNT_IDENTITY_FIELDS
)


class ContextualExpertAggregationArtifactError(ValueError):
    """Raised when an artifact violates the frozen deterministic contract."""


def _stage(value: Any) -> str:
    if type(value) is not str or value not in STAGE_ORDER:
        raise ContextualExpertAggregationArtifactError(
            "stage must be exactly development or confirmation"
        )
    return value


def _strict_keys(value: Any, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContextualExpertAggregationArtifactError(
            f"{field} has missing or unexpected fields"
        )
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContextualExpertAggregationArtifactError(
            "artifact is not canonical finite JSON"
        ) from exc


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ContextualExpertAggregationArtifactError(
            f"{field} must be a lowercase sha256 digest"
        )
    return value


def _canonical_iso_date(value: Any, *, field: str) -> str:
    if type(value) is str:
        if _ISO_DATE_RE.fullmatch(value) is None:
            raise ContextualExpertAggregationArtifactError(
                f"{field} must be an exact ISO date"
            )
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ContextualExpertAggregationArtifactError(
                f"{field} must be an exact ISO date"
            ) from exc
        if parsed.isoformat() != value:
            raise ContextualExpertAggregationArtifactError(
                f"{field} must be a canonical ISO date"
            )
        return value
    if isinstance(value, pd.Timestamp):
        if value.tz is not None or value.time() != time.min:
            raise ContextualExpertAggregationArtifactError(
                f"{field} timestamp must be timezone-naive midnight"
            )
        return value.date().isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is not None or value.time() != time.min:
            raise ContextualExpertAggregationArtifactError(
                f"{field} datetime must be timezone-naive midnight"
            )
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise ContextualExpertAggregationArtifactError(
        f"{field} must be an exact ISO date"
    )


def _normalize_scalar(value: Any, scalar_type: str, *, field: str) -> Any:
    if scalar_type not in SCALAR_TYPES:
        raise ContextualExpertAggregationArtifactError(
            f"{field} declares an unsupported scalar type"
        )
    nullable = scalar_type.startswith("nullable_")
    base_type = scalar_type.removeprefix("nullable_")
    if value is None:
        if nullable:
            return None
        raise ContextualExpertAggregationArtifactError(
            f"{field} may not be null"
        )

    if base_type == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise ContextualExpertAggregationArtifactError(
                f"{field} must be an exact boolean"
            )
        return bool(value)
    if base_type == "int":
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ContextualExpertAggregationArtifactError(
                f"{field} must be an exact integer"
            )
        return int(value)
    if base_type == "float":
        if isinstance(value, (bool, np.bool_, int, np.integer)) or not isinstance(
            value, (float, np.floating)
        ):
            raise ContextualExpertAggregationArtifactError(
                f"{field} must be an exact float"
            )
        result = float(value)
        if not math.isfinite(result):
            raise ContextualExpertAggregationArtifactError(
                f"{field} must be finite"
            )
        return result
    if base_type == "string":
        if type(value) is not str:
            raise ContextualExpertAggregationArtifactError(
                f"{field} must be an exact string"
            )
        return value
    if base_type == "iso_date":
        return _canonical_iso_date(value, field=field)
    raise AssertionError("unreachable scalar type")


@dataclass(frozen=True)
class CanonicalTableSchema:
    columns: tuple[str, ...]
    column_types: tuple[str, ...]
    index_name: str | None = None
    index_type: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.columns
            or len(self.columns) != len(self.column_types)
            or len(set(self.columns)) != len(self.columns)
            or any(type(name) is not str or not name for name in self.columns)
            or any(value not in SCALAR_TYPES for value in self.column_types)
        ):
            raise ContextualExpertAggregationArtifactError(
                "table schema columns or scalar types are invalid"
            )
        if self.index_name is None:
            if self.index_type is not None:
                raise ContextualExpertAggregationArtifactError(
                    "an index type requires an index name"
                )
        elif (
            type(self.index_name) is not str
            or not self.index_name
            or self.index_type not in _NON_NULL_TYPES
        ):
            raise ContextualExpertAggregationArtifactError(
                "table index schema is invalid"
            )


def _table_envelope(frame: pd.DataFrame, schema: CanonicalTableSchema) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != schema.columns:
        raise ContextualExpertAggregationArtifactError(
            "table frame does not have the exact frozen columns"
        )
    rows: list[list[Any]] = []
    for row_position, raw_row in enumerate(
        frame.itertuples(index=False, name=None)
    ):
        rows.append(
            [
                _normalize_scalar(
                    value,
                    scalar_type,
                    field=f"rows[{row_position}][{column}]",
                )
                for column, scalar_type, value in zip(
                    schema.columns, schema.column_types, raw_row
                )
            ]
        )

    index_values: list[Any] | None
    if schema.index_name is None:
        if not isinstance(frame.index, pd.RangeIndex) or not frame.index.equals(
            pd.RangeIndex(len(frame))
        ):
            raise ContextualExpertAggregationArtifactError(
                "an unindexed canonical table requires the default RangeIndex"
            )
        index_values = None
    else:
        if frame.index.name != schema.index_name:
            raise ContextualExpertAggregationArtifactError(
                "table index name differs from the frozen schema"
            )
        assert schema.index_type is not None
        index_values = [
            _normalize_scalar(
                value,
                schema.index_type,
                field=f"index_values[{position}]",
            )
            for position, value in enumerate(frame.index.tolist())
        ]
        if len(set(index_values)) != len(index_values) or any(
            right <= left for left, right in zip(index_values, index_values[1:])
        ):
            raise ContextualExpertAggregationArtifactError(
                "canonical table index must be unique and strictly increasing"
            )

    return {
        "table_schema_version": TABLE_SCHEMA_VERSION,
        "index_name": schema.index_name,
        "index_type": schema.index_type,
        "columns": list(schema.columns),
        "column_types": list(schema.column_types),
        "index_values": index_values,
        "rows": rows,
    }


def canonical_table_bytes(
    frame: pd.DataFrame, *, schema: CanonicalTableSchema
) -> bytes:
    """Serialize one heterogeneous frame without coercing primitive types."""

    if not isinstance(schema, CanonicalTableSchema):
        raise ContextualExpertAggregationArtifactError(
            "canonical table serialization requires a frozen schema"
        )
    return _canonical_json_bytes(_table_envelope(frame, schema))


def parse_canonical_table_bytes(
    payload: bytes, *, schema: CanonicalTableSchema
) -> pd.DataFrame:
    """Parse canonical table bytes and reject every noncanonical representation."""

    if type(payload) is not bytes or not isinstance(schema, CanonicalTableSchema):
        raise ContextualExpertAggregationArtifactError(
            "canonical table parsing requires exact bytes and schema"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContextualExpertAggregationArtifactError(
            "canonical table bytes are not strict JSON"
        ) from exc
    envelope = _strict_keys(value, _TABLE_ENVELOPE_FIELDS, field="table envelope")
    if (
        isinstance(envelope["table_schema_version"], bool)
        or envelope["table_schema_version"] != TABLE_SCHEMA_VERSION
        or envelope["columns"] != list(schema.columns)
        or envelope["column_types"] != list(schema.column_types)
        or envelope["index_name"] != schema.index_name
        or envelope["index_type"] != schema.index_type
        or not isinstance(envelope["rows"], list)
    ):
        raise ContextualExpertAggregationArtifactError(
            "canonical table envelope differs from its frozen schema"
        )
    raw_rows = envelope["rows"]
    normalized_rows: list[list[Any]] = []
    for row_position, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, list) or len(raw_row) != len(schema.columns):
            raise ContextualExpertAggregationArtifactError(
                "canonical table row has the wrong shape"
            )
        normalized_rows.append(
            [
                _normalize_scalar(
                    item,
                    scalar_type,
                    field=f"rows[{row_position}][{column}]",
                )
                for column, scalar_type, item in zip(
                    schema.columns, schema.column_types, raw_row
                )
            ]
        )
    frame = pd.DataFrame(normalized_rows, columns=schema.columns, dtype=object)
    if schema.index_name is None:
        if envelope["index_values"] is not None:
            raise ContextualExpertAggregationArtifactError(
                "unindexed canonical table may not carry index values"
            )
    else:
        raw_index = envelope["index_values"]
        if not isinstance(raw_index, list) or len(raw_index) != len(frame):
            raise ContextualExpertAggregationArtifactError(
                "canonical table index shape is invalid"
            )
        assert schema.index_type is not None
        normalized_index = [
            _normalize_scalar(
                item,
                schema.index_type,
                field=f"index_values[{position}]",
            )
            for position, item in enumerate(raw_index)
        ]
        if schema.index_type == "iso_date":
            frame.index = pd.DatetimeIndex(normalized_index, name=schema.index_name)
        else:
            frame.index = pd.Index(normalized_index, name=schema.index_name)
    if canonical_table_bytes(frame, schema=schema) != payload:
        raise ContextualExpertAggregationArtifactError(
            "table JSON bytes are valid but not canonical"
        )
    return frame


def _payload_inventory(stage: str) -> frozenset[str]:
    selected = _stage(stage)
    token = CUTOFF_TOKEN_BY_STAGE[selected]
    names = {
        ".gitattributes",
        "input_provenance.json",
        "source_bundle_provenance.json",
        f"{selected}_prices_through_{token}.csv",
        f"{selected}_fixed_features.table.json",
        f"{selected}_fixed_parent_prefix_proof.json",
        f"{selected}_forecast__fixed_comparators.table.json",
        f"{selected}_state_weight_diagnostics.table.json",
        f"{selected}_replay_diagnostics.json",
        f"{selected}_pending_lessons.json",
        f"{selected}_checkpoint_through_{token}.json",
        f"{selected}_metrics.json",
        f"{selected}_integrity_evidence.json",
        f"{selected}_gate_report.json",
        f"{selected}_runtime_cost_evidence.json",
        "report.json",
    }
    names.update(
        f"{selected}_forecast__{arm}.table.json" for arm in ARM_ORDER
    )
    names.update(
        f"{selected}_matured_lessons__{arm}.table.json" for arm in ARM_ORDER
    )
    names.update(
        f"{selected}_ledger__{cost}__{policy}.table.json"
        for cost in COST_ORDER
        for policy in POLICY_ORDER
    )
    names.update(f"{selected}_episodes__{cost}.json" for cost in COST_ORDER)
    names.update(f"{selected}_xor__{cost}.json" for cost in COST_ORDER)
    if selected == DEVELOPMENT_STAGE:
        names.add("development_arm_seed_equivalence.json")
    else:
        names.update(
            {
                "development_parent_manifest.json",
                "development_parent_checksums.json",
                "confirmation_attempt_authorization.json",
                "confirmation_prefix_continuity_proof.json",
            }
        )
    return frozenset(names)


DEVELOPMENT_PAYLOAD_NAMES = _payload_inventory(DEVELOPMENT_STAGE)
CONFIRMATION_PAYLOAD_NAMES = _payload_inventory(CONFIRMATION_STAGE)
PAYLOAD_NAMES_BY_STAGE: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        DEVELOPMENT_STAGE: DEVELOPMENT_PAYLOAD_NAMES,
        CONFIRMATION_STAGE: CONFIRMATION_PAYLOAD_NAMES,
    }
)


def payload_names_for_stage(stage: str) -> frozenset[str]:
    return PAYLOAD_NAMES_BY_STAGE[_stage(stage)]


def _as_replay_checkpoint(value: Any, *, field: str) -> ReplayCheckpoint:
    try:
        checkpoint = (
            value
            if isinstance(value, ReplayCheckpoint)
            else ReplayCheckpoint.from_dict(value)
        )
        checkpoint.validate()
    except (TypeError, ValueError, KeyError) as exc:
        raise ContextualExpertAggregationArtifactError(
            f"{field} is not a valid replay checkpoint"
        ) from exc
    return checkpoint


def _as_account_state(value: Any, *, field: str) -> AccountState:
    try:
        return (
            value
            if isinstance(value, AccountState)
            else AccountState.from_checkpoint(value)
        )
    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
        raise ContextualExpertAggregationArtifactError(
            f"{field} is not a valid account checkpoint"
        ) from exc


def _normalize_replay_checkpoints(value: Any) -> dict[str, ReplayCheckpoint]:
    mapping = _strict_keys(value, set(ARM_ORDER), field="replay_checkpoints")
    return {
        arm: _as_replay_checkpoint(mapping[arm], field=f"replay_checkpoints[{arm}]")
        for arm in ARM_ORDER
    }


def _normalize_administrative_accounts(
    value: Any, *, cutoff: str
) -> dict[str, dict[str, AccountState]]:
    costs = _strict_keys(value, set(COST_ORDER), field="administrative_accounts")
    normalized: dict[str, dict[str, AccountState]] = {}
    for cost in COST_ORDER:
        policies = _strict_keys(
            costs[cost], set(POLICY_ORDER), field=f"administrative_accounts[{cost}]"
        )
        normalized[cost] = {}
        for policy in POLICY_ORDER:
            state = _as_account_state(
                policies[policy],
                field=f"administrative_accounts[{cost}][{policy}]",
            )
            if (
                state.policy_name != policy
                or state.cost_bps != COST_BPS[cost]
                or state.inception_count != 1
                or state.last_session_date != cutoff
                or state.pending_decision_date != cutoff
            ):
                raise ContextualExpertAggregationArtifactError(
                    "administrative account policy, cost, inception, or cutoff changed"
                )
            normalized[cost][policy] = state
    return normalized


def _economic_account_payload(state: AccountState) -> dict[str, Any]:
    value = asdict(state)
    return {name: value[name] for name in ECONOMIC_ACCOUNT_FIELDS}


def _require_always_long_benchmark_equivalence(
    accounts: Mapping[str, Mapping[str, AccountState]],
) -> None:
    for cost in COST_ORDER:
        left_state = accounts[cost]["always_long"]
        right_state = accounts[cost]["aapl_buy_hold"]
        left = _economic_account_payload(left_state)
        right = _economic_account_payload(right_state)
        if left != right:
            raise ContextualExpertAggregationArtifactError(
                "always-LONG and buy-and-hold administrative accounts differ"
            )
        for state in (left_state, right_state):
            if (
                state.held_target != 1
                or state.previous_requested_target != 1
                or state.pending_target_exposure != 1
                or state.cash != 0.0
                or state.shares <= 0.0
                or state.last_trade_fill_date != ACCOUNT_INCEPTION_FILL_DATE
                or state.open_cash_entry_decision_date is not None
                or state.open_cash_entry_fill_date is not None
                or state.open_cash_entry_reference_price is not None
                or state.open_cash_fill_observations != 0
                or state.cumulative_active_log_edge != 0.0
            ):
                raise ContextualExpertAggregationArtifactError(
                    "always-LONG and buy-and-hold accounts must remain exactly LONG"
                )


def build_arm_seed_equivalence(
    administrative_accounts: Mapping[str, Mapping[str, AccountState | Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the exact proof that four policy-labelled arm seeds are economic clones."""

    # This helper is deliberately development-specific.
    accounts = _normalize_administrative_accounts(
        administrative_accounts,
        cutoff=CHECKPOINT_CUTOFF_BY_STAGE[DEVELOPMENT_STAGE],
    )
    by_cost: dict[str, Any] = {}
    for cost in COST_ORDER:
        states = {
            arm: _economic_account_payload(accounts[cost][arm]) for arm in ARM_ORDER
        }
        source = states[ONLINE_FULL_ARM]
        if any(states[arm] != source for arm in ARM_ORDER):
            raise ContextualExpertAggregationArtifactError(
                "development arm accounts are not identical online-target seeds"
            )
        by_cost[cost] = {
            "economic_state_sha256": _sha256(source),
            "arm_account_checkpoint_sha256": {
                arm: accounts[cost][arm].to_checkpoint()["account_state_sha256"]
                for arm in ARM_ORDER
            },
            "all_equal": True,
        }
    unsigned = {
        "seed_equivalence_schema_version": SEED_EQUIVALENCE_SCHEMA_VERSION,
        "seed_target_source_policy": ONLINE_FULL_ARM,
        "arm_order": list(ARM_ORDER),
        "cost_order": list(COST_ORDER),
        "economic_fields": list(ECONOMIC_ACCOUNT_FIELDS),
        "excluded_identity_fields": sorted(_ACCOUNT_IDENTITY_FIELDS),
        "by_cost": by_cost,
    }
    return {**unsigned, "proof_sha256": _sha256(unsigned)}


def _expected_runtime(stage: str, arm: str) -> dict[str, Any]:
    if stage == DEVELOPMENT_STAGE:
        return {
            "learning_mode": CAUSAL_ONLINE_MODE,
            "frozen_cutoff": None,
            "ablation_mode": FULL_MODE,
        }
    if arm == ONLINE_FULL_ARM:
        return {
            "learning_mode": CAUSAL_ONLINE_MODE,
            "frozen_cutoff": None,
            "ablation_mode": FULL_MODE,
        }
    if arm == FROZEN_2018_ARM:
        return {
            "learning_mode": FROZEN_CUTOFF_MODE,
            "frozen_cutoff": CHECKPOINT_CUTOFF_BY_STAGE[DEVELOPMENT_STAGE],
            "ablation_mode": FULL_MODE,
        }
    if arm == GLOBAL_ONLY_ARM:
        return {
            "learning_mode": CAUSAL_ONLINE_MODE,
            "frozen_cutoff": None,
            "ablation_mode": GLOBAL_ONLY_MODE,
        }
    if arm == LIFETIME_ONLY_ARM:
        return {
            "learning_mode": CAUSAL_ONLINE_MODE,
            "frozen_cutoff": None,
            "ablation_mode": LIFETIME_ONLY_MODE,
        }
    raise AssertionError("unreachable arm")


def _action_independent_replay_identity(
    checkpoint: ReplayCheckpoint,
) -> dict[str, Any]:
    """Return every replay field that must be identical across forked arms."""

    return {
        "checkpoint_date": checkpoint.checkpoint_date,
        "source_start_date": checkpoint.source_start_date,
        "source_session_count": checkpoint.source_session_count,
        "fixed_feature_columns": list(checkpoint.fixed_feature_columns),
        "market_feature_tail": list(checkpoint.market_feature_tail),
        "signal_cooldown_tail": list(checkpoint.signal_cooldown_tail),
        "cooldown_predecessor": dict(checkpoint.cooldown_predecessor),
        "opportunity_counts": dict(checkpoint.opportunity_counts),
        "opportunity_hashes": dict(checkpoint.opportunity_hashes),
        "market_prefix_sha256": checkpoint.market_prefix_sha256,
        "fixed_feature_prefix_sha256": checkpoint.fixed_feature_prefix_sha256,
        "market_history_tail_sha256": checkpoint.market_history_tail_sha256,
        "signal_cooldown_tail_sha256": checkpoint.signal_cooldown_tail_sha256,
        "model_market_history": checkpoint.model_payload["state"]["market_history"],
    }


def _require_confirmation_runtime_fork_lineage(
    checkpoints: Mapping[str, ReplayCheckpoint],
) -> None:
    online_segments = list(checkpoints[ONLINE_FULL_ARM].runtime_segments)
    expected_transition_lineage: dict[str, Any] | None = None
    for arm in (FROZEN_2018_ARM, GLOBAL_ONLY_ARM, LIFETIME_ONLY_ARM):
        segments = list(checkpoints[arm].runtime_segments)
        if len(segments) != len(online_segments) + 1 or segments[:-1] != online_segments:
            raise ContextualExpertAggregationArtifactError(
                "confirmation arms do not share the exact pre-fork runtime lineage"
            )
        transition = segments[-1]
        lineage = {
            name: transition[name]
            for name in (
                "first_session_date",
                "source_offset",
                "prior_segment_terminal_date",
                "prior_segment_terminal_count",
                "parent_checkpoint_digest",
            )
        }
        if (
            lineage["prior_segment_terminal_date"]
            != LAST_OBSERVED_SESSION_BY_STAGE[DEVELOPMENT_STAGE]
            or lineage["prior_segment_terminal_count"]
            != SOURCE_SESSION_COUNT_BY_STAGE[DEVELOPMENT_STAGE]
            or lineage["source_offset"]
            != SOURCE_SESSION_COUNT_BY_STAGE[DEVELOPMENT_STAGE]
            or lineage["first_session_date"] != FIRST_CONFIRMATION_SESSION
        ):
            raise ContextualExpertAggregationArtifactError(
                "confirmation fork does not begin from the exact development cutoff"
            )
        if expected_transition_lineage is None:
            expected_transition_lineage = lineage
        elif lineage != expected_transition_lineage:
            raise ContextualExpertAggregationArtifactError(
                "confirmation arms do not share the exact fork transition"
            )


def _validate_replay_stage_semantics(
    stage: str, checkpoints: Mapping[str, ReplayCheckpoint]
) -> None:
    last_observed = LAST_OBSERVED_SESSION_BY_STAGE[stage]
    action_independent_identity: dict[str, Any] | None = None
    for arm in ARM_ORDER:
        checkpoint = checkpoints[arm]
        runtime = checkpoint.model_payload["runtime"]
        if (
            checkpoint.checkpoint_date != last_observed
            or checkpoint.source_start_date != SOURCE_START_SESSION_BY_STAGE[stage]
            or checkpoint.source_session_count != SOURCE_SESSION_COUNT_BY_STAGE[stage]
            or runtime != _expected_runtime(stage, arm)
        ):
            raise ContextualExpertAggregationArtifactError(
                "replay checkpoint source bounds, cutoff, or terminal runtime changed"
            )
        identity = _action_independent_replay_identity(checkpoint)
        if action_independent_identity is None:
            action_independent_identity = identity
        elif identity != action_independent_identity:
            raise ContextualExpertAggregationArtifactError(
                "replay arms do not share exact action-independent source identity"
            )
    if stage == DEVELOPMENT_STAGE:
        online = checkpoints[ONLINE_FULL_ARM].to_dict()
        if any(checkpoints[arm].to_dict() != online for arm in ARM_ORDER):
            raise ContextualExpertAggregationArtifactError(
                "development replay arm seeds must be the identical online checkpoint"
            )
    else:
        _require_confirmation_runtime_fork_lineage(checkpoints)
        online_state = checkpoints[ONLINE_FULL_ARM].model_payload["state"]
        if any(
            checkpoints[arm].model_payload["state"] != online_state
            for arm in (GLOBAL_ONLY_ARM, LIFETIME_ONLY_ARM)
        ):
            raise ContextualExpertAggregationArtifactError(
                "confirmation action ablations changed action-independent state"
            )


_COMPOSITE_FIELDS = {
    "contract_version",
    "checkpoint_schema_version",
    "stage",
    "checkpoint_cutoff",
    "last_observed_session",
    "arm_order",
    "policy_order",
    "cost_order",
    "replay_checkpoints",
    "administrative_accounts",
    "arm_seed_equivalence",
    "checkpoint_sha256",
}


@dataclass(frozen=True)
class CompositeStageCheckpoint:
    contract_version: str
    checkpoint_schema_version: int
    stage: str
    checkpoint_cutoff: str
    last_observed_session: str
    replay_checkpoints: Mapping[str, ReplayCheckpoint]
    administrative_accounts: Mapping[str, Mapping[str, AccountState]]
    arm_seed_equivalence: Mapping[str, Any] | None
    checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "stage": self.stage,
            "checkpoint_cutoff": self.checkpoint_cutoff,
            "last_observed_session": self.last_observed_session,
            "arm_order": list(ARM_ORDER),
            "policy_order": list(POLICY_ORDER),
            "cost_order": list(COST_ORDER),
            "replay_checkpoints": {
                arm: self.replay_checkpoints[arm].to_dict() for arm in ARM_ORDER
            },
            "administrative_accounts": {
                cost: {
                    policy: self.administrative_accounts[cost][policy].to_checkpoint()
                    for policy in POLICY_ORDER
                }
                for cost in COST_ORDER
            },
            "arm_seed_equivalence": (
                None
                if self.arm_seed_equivalence is None
                else dict(self.arm_seed_equivalence)
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def build_composite_stage_checkpoint(
    *,
    stage: str,
    replay_checkpoints: Mapping[str, ReplayCheckpoint | Mapping[str, Any]],
    administrative_accounts: Mapping[
        str, Mapping[str, AccountState | Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    """Construct and self-hash one exact model plus administrative checkpoint."""

    selected = _stage(stage)
    cutoff = CHECKPOINT_CUTOFF_BY_STAGE[selected]
    last_observed = LAST_OBSERVED_SESSION_BY_STAGE[selected]
    replays = _normalize_replay_checkpoints(replay_checkpoints)
    _validate_replay_stage_semantics(selected, replays)
    accounts = _normalize_administrative_accounts(
        administrative_accounts, cutoff=last_observed
    )
    _require_always_long_benchmark_equivalence(accounts)
    seed_proof = (
        build_arm_seed_equivalence(accounts)
        if selected == DEVELOPMENT_STAGE
        else None
    )
    unsigned = {
        "contract_version": CONTRACT_VERSION,
        "checkpoint_schema_version": COMPOSITE_CHECKPOINT_SCHEMA_VERSION,
        "stage": selected,
        "checkpoint_cutoff": cutoff,
        "last_observed_session": last_observed,
        "arm_order": list(ARM_ORDER),
        "policy_order": list(POLICY_ORDER),
        "cost_order": list(COST_ORDER),
        "replay_checkpoints": {
            arm: replays[arm].to_dict() for arm in ARM_ORDER
        },
        "administrative_accounts": {
            cost: {
                policy: accounts[cost][policy].to_checkpoint()
                for policy in POLICY_ORDER
            }
            for cost in COST_ORDER
        },
        "arm_seed_equivalence": seed_proof,
    }
    return {**unsigned, "checkpoint_sha256": _sha256(unsigned)}


def parse_composite_stage_checkpoint(value: Any) -> CompositeStageCheckpoint:
    """Parse a composite checkpoint and regenerate every semantic proof."""

    payload = _strict_keys(value, _COMPOSITE_FIELDS, field="composite checkpoint")
    if (
        payload["contract_version"] != CONTRACT_VERSION
        or isinstance(payload["checkpoint_schema_version"], bool)
        or payload["checkpoint_schema_version"]
        != COMPOSITE_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ContextualExpertAggregationArtifactError(
            "composite checkpoint contract or schema version changed"
        )
    selected = _stage(payload["stage"])
    cutoff = CHECKPOINT_CUTOFF_BY_STAGE[selected]
    last_observed = LAST_OBSERVED_SESSION_BY_STAGE[selected]
    if (
        payload["checkpoint_cutoff"] != cutoff
        or payload["last_observed_session"] != last_observed
        or payload["arm_order"] != list(ARM_ORDER)
        or payload["policy_order"] != list(POLICY_ORDER)
        or payload["cost_order"] != list(COST_ORDER)
    ):
        raise ContextualExpertAggregationArtifactError(
            "composite checkpoint cutoff or frozen ordering changed"
        )
    supplied_hash = _require_sha256(
        payload["checkpoint_sha256"], field="checkpoint_sha256"
    )
    unsigned = {name: payload[name] for name in _COMPOSITE_FIELDS - {"checkpoint_sha256"}}
    if supplied_hash != _sha256(unsigned):
        raise ContextualExpertAggregationArtifactError(
            "composite checkpoint self-hash is invalid"
        )
    expected = build_composite_stage_checkpoint(
        stage=selected,
        replay_checkpoints=payload["replay_checkpoints"],
        administrative_accounts=payload["administrative_accounts"],
    )
    if _canonical_json_bytes(dict(payload)) != _canonical_json_bytes(expected):
        raise ContextualExpertAggregationArtifactError(
            "composite checkpoint differs from deterministic regeneration"
        )
    replays = _normalize_replay_checkpoints(expected["replay_checkpoints"])
    accounts = _normalize_administrative_accounts(
        expected["administrative_accounts"], cutoff=last_observed
    )
    return CompositeStageCheckpoint(
        contract_version=CONTRACT_VERSION,
        checkpoint_schema_version=COMPOSITE_CHECKPOINT_SCHEMA_VERSION,
        stage=selected,
        checkpoint_cutoff=cutoff,
        last_observed_session=last_observed,
        replay_checkpoints=replays,
        administrative_accounts=accounts,
        arm_seed_equivalence=expected["arm_seed_equivalence"],
        checkpoint_sha256=expected["checkpoint_sha256"],
    )


def composite_stage_checkpoint_bytes(value: Any) -> bytes:
    parsed = parse_composite_stage_checkpoint(value)
    return _canonical_json_bytes(parsed.to_dict())


def parse_composite_stage_checkpoint_bytes(payload: bytes) -> CompositeStageCheckpoint:
    if type(payload) is not bytes:
        raise ContextualExpertAggregationArtifactError(
            "composite checkpoint must be exact bytes"
        )
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContextualExpertAggregationArtifactError(
            "composite checkpoint bytes are not strict JSON"
        ) from exc
    parsed = parse_composite_stage_checkpoint(value)
    if _canonical_json_bytes(parsed.to_dict()) != payload:
        raise ContextualExpertAggregationArtifactError(
            "composite checkpoint JSON is not canonical"
        )
    return parsed


__all__ = [
    "ARM_ORDER",
    "CHECKPOINT_CUTOFF_BY_STAGE",
    "COMPOSITE_CHECKPOINT_SCHEMA_VERSION",
    "CONFIRMATION_PAYLOAD_NAMES",
    "CONFIRMATION_STAGE",
    "CONTRACT_VERSION",
    "COST_BPS",
    "COST_ORDER",
    "CanonicalTableSchema",
    "CompositeStageCheckpoint",
    "ContextualExpertAggregationArtifactError",
    "DEVELOPMENT_PAYLOAD_NAMES",
    "DEVELOPMENT_STAGE",
    "ECONOMIC_ACCOUNT_FIELDS",
    "FIXED_POLICY_ORDER",
    "FIRST_CONFIRMATION_SESSION",
    "GIT_ATTRIBUTES_BYTES",
    "LAST_OBSERVED_SESSION_BY_STAGE",
    "OUTPUT_DIRECTORY_BY_STAGE",
    "OUTPUT_PARENT",
    "PAYLOAD_NAMES_BY_STAGE",
    "POLICY_ORDER",
    "RUN_ID_BY_STAGE",
    "SCALAR_TYPES",
    "STAGE_ORDER",
    "SOURCE_SESSION_COUNT_BY_STAGE",
    "SOURCE_START_SESSION_BY_STAGE",
    "TABLE_SCHEMA_VERSION",
    "build_arm_seed_equivalence",
    "build_composite_stage_checkpoint",
    "canonical_table_bytes",
    "composite_stage_checkpoint_bytes",
    "parse_canonical_table_bytes",
    "parse_composite_stage_checkpoint",
    "parse_composite_stage_checkpoint_bytes",
    "payload_names_for_stage",
]
