"""Pure artifact contract for the frozen-policy 2024 audit.

This module owns deterministic names, schemas, canonical JSON/table bytes and
seal metadata.  It deliberately performs no data acquisition, replay, policy
decision, scoring, or economic evaluation.  Promotion and the one-shot marker
commit protocol remain runner responsibilities.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from . import contextual_expert_aggregation_artifacts as _base_artifacts
from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_stage as _stage_artifacts


CONTRACT_VERSION = "aapl-causal-contextual-expert-aggregation-2024-audit-v2"
EXPECTED_BRANCH = "codex/aapl-causal-contextual-expert-aggregation-2024-audit-v2"
AUDIT_STAGE = "audit_2024"
AUDIT_RUN_ID = "contextual-expert-aggregation-frozen-policy-2024-audit-v2"
VERIFIER_ID = "contextual-expert-aggregation-2024-audit-verifier-v2"
PREREGISTRATION_COMMIT = "2f4537440eccb41c62ab553d2ffb76460296cc9d"

CONTROL_ROOT = Path("e/aapl_causal_contextual_expert_aggregation_2024_audit_v2")
OUTPUT_PARENT = CONTROL_ROOT / "runs"
OUTPUT_DIRECTORY = OUTPUT_PARENT / AUDIT_RUN_ID
PRIVATE_DIRECTORY = OUTPUT_PARENT / f".sealing-{AUDIT_RUN_ID}"
FAILED_DIRECTORY = OUTPUT_PARENT / f".failed-{AUDIT_RUN_ID}"
ATTEMPT_LOCK_FILENAME = "AUDIT_2024_ATTEMPT_LOCK.json"
SUCCESS_MARKER_FILENAME = "AUDIT_2024_STAGE_SUCCESS.json"
PENDING_SUCCESS_MARKER_FILENAME = "AUDIT_2024_STAGE_SUCCESS.pending"
ATTEMPT_LOCK_PATH = CONTROL_ROOT / ATTEMPT_LOCK_FILENAME
SUCCESS_MARKER_PATH = CONTROL_ROOT / SUCCESS_MARKER_FILENAME
PENDING_SUCCESS_MARKER_PATH = CONTROL_ROOT / PENDING_SUCCESS_MARKER_FILENAME

GIT_ATTRIBUTES_BYTES = b"* -text\n"
GIT_ATTRIBUTES_SHA256 = (
    "sha256:705fd4d6451a31d36b3df7de96f83f30ac976c9b4a6d1e51671d8e2f33e2d0da"
)
STAGE_DEADLINE_SECONDS = 1_800.0
FINALIZATION_RESERVE_SECONDS = 5.0
TABLE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
MARKER_SCHEMA_VERSION = 1

PARENT_MANIFEST_FILE_SHA256 = (
    "sha256:fd4c7427bdc9ec2a5463dcc2781310aeab5c498836db0617c0c41ebf81267bbd"
)
PARENT_MANIFEST_SELF_SHA256 = (
    "sha256:c71e201a8a3d1ef2b0ebe914d46abfdd0f5a9154d4f106ce723168a9bce009e8"
)
PARENT_CHECKPOINT_FILE_SHA256 = (
    "sha256:560571500f580f9e0fd4319a93905b4ad1e498f1c1f27de9d647da455b6d5477"
)
PARENT_CHECKPOINT_SELF_SHA256 = (
    "sha256:4aaf549c9352ac322f36c46cb8dcdd03f672a1eba580813ba8a1947d323a85eb"
)

COST_ORDER = ("base_5bps", "stress_10bps")
SCENARIO_ORDER = ("frozen_2023_lead", "online_2024_shadow")
FIXED_POLICY_ORDER = (
    "aapl_buy_hold",
    "always_long",
    "exact_union_cash",
    "contextual_only",
)
POLICY_ORDER = (*SCENARIO_ORDER, *FIXED_POLICY_ORDER)

BUNDLE_FILE_ORDER = (
    ".gitattributes",
    ATTEMPT_LOCK_FILENAME,
    "input_snapshot_receipt.json",
    "audit_prices_through_2024.csv",
    "parent_stage_manifest.json",
    "parent_checksums.json",
    "parent_verification.json",
    "prefix_continuity_proof.json",
    "source_bundle_provenance.json",
    "known_result_isolation_evidence.json",
    "audit_fixed_features_2024.table.json",
    "audit_forecast__frozen_2023_lead.table.json",
    "audit_forecast__online_2024_shadow.table.json",
    "audit_forecast__fixed_comparators.table.json",
    "audit_matured_lessons__frozen_2023_lead.table.json",
    "audit_matured_lessons__online_2024_shadow.table.json",
    "audit_state_weight_diagnostics_2024.table.json",
    "audit_continuation_checkpoint_through_2024.json",
    "audit_pending_lessons.json",
    "audit_replay_diagnostics.json",
    "audit_ledger__base_5bps__frozen_2023_lead.table.json",
    "audit_ledger__base_5bps__online_2024_shadow.table.json",
    "audit_ledger__base_5bps__aapl_buy_hold.table.json",
    "audit_ledger__base_5bps__always_long.table.json",
    "audit_ledger__base_5bps__exact_union_cash.table.json",
    "audit_ledger__base_5bps__contextual_only.table.json",
    "audit_ledger__stress_10bps__frozen_2023_lead.table.json",
    "audit_ledger__stress_10bps__online_2024_shadow.table.json",
    "audit_ledger__stress_10bps__aapl_buy_hold.table.json",
    "audit_ledger__stress_10bps__always_long.table.json",
    "audit_ledger__stress_10bps__exact_union_cash.table.json",
    "audit_ledger__stress_10bps__contextual_only.table.json",
    "audit_episodes__base_5bps.json",
    "audit_episodes__stress_10bps.json",
    "audit_xor__base_5bps.json",
    "audit_xor__stress_10bps.json",
    "audit_metrics.json",
    "audit_gate_report.json",
    "audit_integrity_evidence.json",
    "audit_runtime_cost_evidence.json",
    "report.json",
    "checksums.json",
    "stage_manifest.json",
)
PAYLOAD_FILE_ORDER = BUNDLE_FILE_ORDER[:41]
CHECKSUM_FILE_ORDER = (*PAYLOAD_FILE_ORDER, "stage_manifest.json")
BUNDLE_FILENAMES = frozenset(BUNDLE_FILE_ORDER)
PAYLOAD_FILENAMES = frozenset(PAYLOAD_FILE_ORDER)
CHECKSUM_FILENAMES = frozenset(CHECKSUM_FILE_ORDER)

if (
    len(BUNDLE_FILE_ORDER) != 43
    or len(BUNDLE_FILENAMES) != 43
    or len(PAYLOAD_FILE_ORDER) != 41
    or len(CHECKSUM_FILE_ORDER) != 42
    or "checksums.json" in CHECKSUM_FILENAMES
    or SUCCESS_MARKER_FILENAME in BUNDLE_FILENAMES
    or PENDING_SUCCESS_MARKER_FILENAME in BUNDLE_FILENAMES
):
    raise RuntimeError("the frozen 2024 audit inventory is internally inconsistent")


class ContextualExpertAggregation2024AuditArtifactError(ValueError):
    """Raised when a 2024 audit artifact violates its frozen contract."""


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
        raise ContextualExpertAggregation2024AuditArtifactError(
            "artifact is not canonical finite JSON"
        ) from exc


def canonical_json_line_bytes(value: Any) -> bytes:
    """Return canonical compact UTF-8 JSON with exactly one terminal LF."""

    return _canonical_json_bytes(value) + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "hash input must be exact bytes"
        )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


if _sha256_bytes(GIT_ATTRIBUTES_BYTES) != GIT_ATTRIBUTES_SHA256:
    raise RuntimeError("the frozen .gitattributes byte identity changed")


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ContextualExpertAggregation2024AuditArtifactError(
            f"{field} must be a lowercase sha256 digest"
        )
    return value


def _require_commit(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContextualExpertAggregation2024AuditArtifactError(
            f"{field} must be a lowercase full Git commit"
        )
    return value


def _strict_mapping(
    value: Any, expected: set[str] | frozenset[str], *, field: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ContextualExpertAggregation2024AuditArtifactError(
            f"{field} has the wrong exact field inventory"
        )
    return value


def parse_canonical_json_line(payload: bytes, *, field: str = "artifact") -> Any:
    """Parse canonical finite JSON and reject whitespace or duplicate keys."""

    if type(payload) is not bytes or not payload.endswith(b"\n"):
        raise ContextualExpertAggregation2024AuditArtifactError(
            f"{field} must be canonical JSON with one terminal LF"
        )

    def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=exact_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContextualExpertAggregation2024AuditArtifactError(
            f"{field} is not strict JSON"
        ) from exc
    if canonical_json_line_bytes(value) != payload:
        raise ContextualExpertAggregation2024AuditArtifactError(
            f"{field} JSON bytes are valid but not canonical"
        )
    return value


FIXED_FEATURE_SCHEMA = _stage_artifacts.FIXED_FEATURE_SCHEMA
FORECAST_SCHEMA = _stage_artifacts.FORECAST_SCHEMA
MATURED_EVENT_SCHEMA = _stage_artifacts.MATURED_EVENT_SCHEMA
DIAGNOSTIC_SCHEMA = _stage_artifacts.DIAGNOSTIC_SCHEMA
LEDGER_SCHEMA = _stage_artifacts.LEDGER_SCHEMA
FIXED_COMPARATOR_SCHEMA = _base_artifacts.CanonicalTableSchema(
    columns=(
        "fixed_always_long_target_exposure",
        "fixed_union_cash_target_exposure",
        "fixed_contextual_only_target_exposure",
    ),
    column_types=("float", "float", "float"),
    index_name="decision_date",
    index_type="iso_date",
)


def column_list_sha256(
    schema_or_columns: _base_artifacts.CanonicalTableSchema | tuple[str, ...],
) -> str:
    columns = (
        schema_or_columns.columns
        if isinstance(schema_or_columns, _base_artifacts.CanonicalTableSchema)
        else tuple(schema_or_columns)
    )
    if not columns or any(type(value) is not str for value in columns):
        raise ContextualExpertAggregation2024AuditArtifactError(
            "column hash requires a nonempty ordered string inventory"
        )
    return hashlib.sha256(_canonical_json_bytes(list(columns))).hexdigest()


TABLE_COLUMN_SHA256 = MappingProxyType(
    {
        "fixed_features": "8282e3a6cd1ef04f26bc2131d9ec9827a3f0aaa706f5f68fd00cae7ff8c42a20",
        "forecast": "d08661dc9ea8526c9697750143fa061cba2c197641f6c8480fa25c7a512ea9b1",
        "fixed_comparators": "0c6815db9e6e2e7e550e909611129b75ed2329783d2609ed35b561cc241615a2",
        "matured_lessons": "10cf64eea698fa0761acbaa56793bb92ddcf3b50098b21832cae06b43e8afb4c",
        "diagnostics": "558403da0eccf05db0313c8668840f9726ea2ad1652ff0695a24eafa052ba4e0",
        "ledger": "9f1f1b34d3f0db6cedf9b7cbd59ba9c75b8e2be2055e761568397e60fd386971",
    }
)

_SCHEMA_BY_ROLE = {
    "fixed_features": FIXED_FEATURE_SCHEMA,
    "forecast": FORECAST_SCHEMA,
    "fixed_comparators": FIXED_COMPARATOR_SCHEMA,
    "matured_lessons": MATURED_EVENT_SCHEMA,
    "diagnostics": DIAGNOSTIC_SCHEMA,
    "ledger": LEDGER_SCHEMA,
}
if any(
    column_list_sha256(_SCHEMA_BY_ROLE[role]) != expected
    for role, expected in TABLE_COLUMN_SHA256.items()
):
    raise RuntimeError("a reused canonical table schema changed")

_table_schema_by_filename: dict[str, _base_artifacts.CanonicalTableSchema] = {
    "audit_fixed_features_2024.table.json": FIXED_FEATURE_SCHEMA,
    "audit_forecast__frozen_2023_lead.table.json": FORECAST_SCHEMA,
    "audit_forecast__online_2024_shadow.table.json": FORECAST_SCHEMA,
    "audit_forecast__fixed_comparators.table.json": FIXED_COMPARATOR_SCHEMA,
    "audit_matured_lessons__frozen_2023_lead.table.json": MATURED_EVENT_SCHEMA,
    "audit_matured_lessons__online_2024_shadow.table.json": MATURED_EVENT_SCHEMA,
    "audit_state_weight_diagnostics_2024.table.json": DIAGNOSTIC_SCHEMA,
}
for cost in COST_ORDER:
    for policy in POLICY_ORDER:
        _table_schema_by_filename[
            f"audit_ledger__{cost}__{policy}.table.json"
        ] = LEDGER_SCHEMA
TABLE_SCHEMA_BY_FILENAME: Mapping[str, _base_artifacts.CanonicalTableSchema] = (
    MappingProxyType(_table_schema_by_filename)
)

_table_rows_by_filename: dict[str, int | None] = {
    "audit_fixed_features_2024.table.json": 252,
    "audit_forecast__frozen_2023_lead.table.json": 252,
    "audit_forecast__online_2024_shadow.table.json": 252,
    "audit_forecast__fixed_comparators.table.json": 252,
    "audit_matured_lessons__frozen_2023_lead.table.json": None,
    "audit_matured_lessons__online_2024_shadow.table.json": None,
    "audit_state_weight_diagnostics_2024.table.json": 504,
}
for cost in COST_ORDER:
    for policy in POLICY_ORDER:
        _table_rows_by_filename[
            f"audit_ledger__{cost}__{policy}.table.json"
        ] = 5_033
TABLE_EXPECTED_ROWS_BY_FILENAME: Mapping[str, int | None] = MappingProxyType(
    _table_rows_by_filename
)


def table_schema_for_filename(filename: str) -> _base_artifacts.CanonicalTableSchema:
    if type(filename) is not str or filename not in TABLE_SCHEMA_BY_FILENAME:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "unknown canonical table filename"
        )
    return TABLE_SCHEMA_BY_FILENAME[filename]


def canonical_table_bytes(
    frame: pd.DataFrame, *, schema: _base_artifacts.CanonicalTableSchema
) -> bytes:
    # The inherited table envelope is the proven heterogeneous serializer, but
    # this audit's byte contract additionally requires one terminal LF for
    # every JSON artifact.
    return _base_artifacts.canonical_table_bytes(frame, schema=schema) + b"\n"


def parse_canonical_table_bytes(
    payload: bytes, *, schema: _base_artifacts.CanonicalTableSchema
) -> pd.DataFrame:
    if type(payload) is not bytes or not payload.endswith(b"\n"):
        raise ContextualExpertAggregation2024AuditArtifactError(
            "canonical table JSON must have exactly one terminal LF"
        )
    try:
        frame = _base_artifacts.parse_canonical_table_bytes(
            payload[:-1], schema=schema
        )
    except _base_artifacts.ContextualExpertAggregationArtifactError as exc:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "canonical table JSON violates its frozen schema"
        ) from exc
    if canonical_table_bytes(frame, schema=schema) != payload:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "canonical table JSON bytes are not canonical"
        )
    return frame


def canonical_registered_table_bytes(filename: str, frame: pd.DataFrame) -> bytes:
    schema = table_schema_for_filename(filename)
    expected_rows = TABLE_EXPECTED_ROWS_BY_FILENAME[filename]
    if expected_rows is not None and len(frame) != expected_rows:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "canonical table row count differs from the frozen contract"
        )
    if schema.index_name == "decision_date":
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ContextualExpertAggregation2024AuditArtifactError(
                "a registered decision table requires a DatetimeIndex"
            )
        normalized = frame.rename_axis("decision_date")
    elif schema.index_name is None:
        normalized = frame
    else:  # pragma: no cover - the frozen registry has no other index role.
        raise ContextualExpertAggregation2024AuditArtifactError(
            "registered table declares an unsupported index role"
        )
    # Reuse the parent's proven serializers before applying this audit
    # contract's additional terminal-LF byte rule.  Ledgers have one inherited
    # in-memory empty-string sentinel which must canonically become JSON null.
    if schema is LEDGER_SCHEMA:
        return _stage_artifacts.ledger_table_bytes(normalized) + b"\n"
    return _stage_artifacts.table_bytes(normalized, schema=schema) + b"\n"


def parse_registered_table_bytes(filename: str, payload: bytes) -> pd.DataFrame:
    schema = table_schema_for_filename(filename)
    frame = parse_canonical_table_bytes(payload, schema=schema)
    expected_rows = TABLE_EXPECTED_ROWS_BY_FILENAME[filename]
    if expected_rows is not None and len(frame) != expected_rows:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "canonical table row count differs from the frozen contract"
        )
    return frame


def _schema_descriptor(
    schema: _base_artifacts.CanonicalTableSchema, *, rows: int | None
) -> dict[str, Any]:
    return {
        "kind": "canonical_table",
        "table_schema_version": TABLE_SCHEMA_VERSION,
        "envelope_fields": sorted(
            (
                "column_types",
                "columns",
                "index_name",
                "index_type",
                "index_values",
                "rows",
                "table_schema_version",
            )
        ),
        "columns": list(schema.columns),
        "column_types": list(schema.column_types),
        "index_name": schema.index_name,
        "index_type": schema.index_type,
        "expected_rows": rows,
        "column_list_sha256": column_list_sha256(schema),
    }


_MANIFEST_FIELDS = frozenset(
    {
        "manifest_schema_version",
        "manifest_sha256",
        "contract_version",
        "verifier_id",
        "run_id",
        "stage",
        "status",
        "stage_pass",
        "evidence_classification",
        "preregistration_commit",
        "git_identity",
        "dependency_identity_sha256",
        "artifact_schema_registry_sha256",
        "parent_manifest_file_sha256",
        "parent_manifest_self_sha256",
        "parent_checkpoint_file_sha256",
        "parent_checkpoint_self_sha256",
        "receipt_file_sha256",
        "input_raw_sha256",
        "input_canonical_sha256",
        "attempt_lock_file_sha256",
        "payload_sha256",
        "runtime_cost_evidence_file_sha256",
        "gate_report_file_sha256",
        "learning_classification",
        "later_market_data_accessed",
        "prior_2024_artifact_accessed",
        "external_cost_usd",
    }
)
_MANIFEST_UNSIGNED_FIELDS = _MANIFEST_FIELDS - {"manifest_sha256"}

_MARKER_FIELDS = frozenset(
    {
        "marker_schema_version",
        "marker_sha256",
        "contract_version",
        "run_id",
        "stage",
        "preregistration_commit",
        "git_commit",
        "attempt_lock_file_sha256",
        "final_relative_path",
        "final_inventory_sha256",
        "stage_manifest_file_sha256",
        "stage_manifest_self_sha256",
        "checksums_file_sha256",
        "marker_preparation_elapsed_seconds",
        "stage_deadline_seconds",
        "marker_preparation_deadline_pass",
        "external_cost_usd",
    }
)
_MARKER_UNSIGNED_FIELDS = _MARKER_FIELDS - {"marker_sha256"}

_JSON_ARTIFACT_ROLES = {
    "parent_verification.json": "parent_identity_and_selected_continuation",
    "prefix_continuity_proof.json": "canonical_prefix_and_checkpoint_continuity",
    "source_bundle_provenance.json": "authorized_source_lineage",
    "known_result_isolation_evidence.json": "known_result_access_firewall",
    "audit_continuation_checkpoint_through_2024.json": "terminal_model_and_accounts",
    "audit_pending_lessons.json": "pending_and_cooldown_state",
    "audit_replay_diagnostics.json": "causal_replay_diagnostics",
    "audit_episodes__base_5bps.json": "cash_episode_extraction",
    "audit_episodes__stress_10bps.json": "cash_episode_extraction",
    "audit_xor__base_5bps.json": "fill_based_signed_xor_extraction",
    "audit_xor__stress_10bps.json": "fill_based_signed_xor_extraction",
    "audit_metrics.json": "unrounded_economic_diagnostics",
    "audit_gate_report.json": "unrounded_preregistered_gates",
    "audit_integrity_evidence.json": "fatal_integrity_evidence",
    "audit_runtime_cost_evidence.json": "runtime_and_zero_cost_attestation",
    "report.json": "human_and_machine_stage_report",
}

def _object_schema(
    fields: Mapping[str, Any], *, self_hash_field: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "fields": dict(fields)}
    if self_hash_field is not None:
        result["self_hash_field"] = self_hash_field
    return result


def _fixed_map_schema(keys: tuple[str, ...] | list[str], values: Any) -> dict[str, Any]:
    return {"type": "fixed_map", "keys": list(keys), "values": values}


def _list_schema(items: Any, *, length: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "list", "items": items}
    if length is not None:
        result["length"] = length
    return result


def _enum_schema(*values: Any) -> dict[str, Any]:
    return {"type": "enum", "values": list(values)}


def _literal_schema(value: Any) -> dict[str, Any]:
    return {"type": "literal", "value": value}


def _nullable_schema(schema: Any) -> dict[str, Any]:
    return {"type": "nullable", "schema": schema}


_RUNTIME_CONTROL_SCHEMA = _object_schema(
    {
        "learning_mode": _enum_schema("causal_online", "frozen_cutoff"),
        "frozen_cutoff": _nullable_schema("iso_date"),
        "ablation_mode": _enum_schema("full", "global_only", "lifetime_only"),
    }
)

_EXPERT_KEYS = (
    "always_long",
    "union_cash",
    "contextual_only",
    "weak_trend_only",
)
_MARKET_STATE_KEYS = (
    "fast_off_slow_off",
    "fast_off_slow_on",
    "fast_on_slow_off",
    "fast_on_slow_on",
)
_SCALE_KEYS = ("half_life_8", "half_life_32", "lifetime")
_OPPORTUNITY_KEYS = (
    "contextual_virtual_signal",
    "weak_trend_virtual_signal",
    "unfiltered_union_candidate_signal",
    "unfiltered_union_signal",
)
_COOLDOWN_SIGNAL_KEYS = (
    "contextual_raw_signal",
    "contextual_virtual_signal",
    "weak_trend_raw_signal",
    "weak_trend_virtual_signal",
    "unfiltered_union_candidate_signal",
    "unfiltered_union_signal",
)
_COOLDOWN_PREDECESSOR_KEYS = (
    "contextual_virtual_signal",
    "weak_trend_virtual_signal",
    "unfiltered_union_signal",
)

_SIGNAL_MARKET_CONTEXT_SCHEMA = _object_schema(
    {
        "spy_adj_close": "float",
        "qqq_adj_close": "float",
        "spy_adj_close_10": "float",
        "qqq_adj_close_10": "float",
        "spy_adj_close_20": "float",
        "qqq_adj_close_20": "float",
    }
)
_PENDING_LESSON_SCHEMA = _object_schema(
    {
        "signal_date": "iso_date",
        "sessions_until_maturity": "int",
        "entry_adjusted_open": _nullable_schema("float"),
        "market_state": _nullable_schema("string"),
        "market_context": _nullable_schema(_SIGNAL_MARKET_CONTEXT_SCHEMA),
        "expert_cash_advice": _fixed_map_schema(_EXPERT_KEYS, "bool"),
        "learning_eligible": "bool",
    }
)
_MARKET_SESSION_SCHEMA = _object_schema(
    {
        "session_date": "iso_date",
        "aapl_open": "float",
        "aapl_close": "float",
        "aapl_adj_close": "float",
        "aapl_adj_open": "float",
        "spy_adj_close": "float",
        "qqq_adj_close": "float",
        "contextual_signal": "bool",
        "weak_trend_signal": "bool",
        "canonical_union_opportunity": "bool",
    }
)
_POOL_SCHEMA = _object_schema(
    {
        "cumulative_rewards": _fixed_map_schema(_EXPERT_KEYS, "float"),
        "range_energy": "float",
        "effective_count": "float",
    }
)
_MODEL_CONSTANTS_SCHEMA = _object_schema(
    {
        "expert_order": _list_schema("string", length=len(_EXPERT_KEYS)),
        "market_state_order": _list_schema(
            "string", length=len(_MARKET_STATE_KEYS)
        ),
        "fast_lookback": "int",
        "slow_lookback": "int",
        "scale_order": _list_schema("string", length=len(_SCALE_KEYS)),
        "scale_half_lives": _object_schema(
            {
                "half_life_8": "float",
                "half_life_32": "float",
                "lifetime": "null",
            }
        ),
        "per_side_friction": "float",
        "round_trip_log_friction": "float",
        "state_prior_strength": "float",
        "long_tie_tolerance": "float",
        "lesson_maturity_sessions": "int",
        "lesson_memory_start": "iso_date",
    }
)
_MODEL_STATE_SCHEMA = _object_schema(
    {
        "last_session_date": _nullable_schema("iso_date"),
        "processed_session_count": "int",
        "matured_lesson_count": "int",
        "admitted_lesson_count": "int",
        "market_history": _list_schema(_MARKET_SESSION_SCHEMA),
        "pending_lessons": _list_schema(_PENDING_LESSON_SCHEMA),
        "global_pools": _fixed_map_schema(_SCALE_KEYS, _POOL_SCHEMA),
        "state_pools": _fixed_map_schema(
            _SCALE_KEYS, _fixed_map_schema(_MARKET_STATE_KEYS, _POOL_SCHEMA)
        ),
    }
)
_MODEL_PAYLOAD_SCHEMA = _object_schema(
    {
        "serialization_version": "int",
        "method": "string",
        "constants": _MODEL_CONSTANTS_SCHEMA,
        "runtime": _RUNTIME_CONTROL_SCHEMA,
        "state": _MODEL_STATE_SCHEMA,
    }
)
_RUNTIME_SEGMENT_SCHEMA = _object_schema(
    {
        "first_session_date": "iso_date",
        "source_offset": "int",
        "runtime": _RUNTIME_CONTROL_SCHEMA,
        "prior_segment_terminal_date": _nullable_schema("iso_date"),
        "prior_segment_terminal_count": "int",
        # Inherited replay digests are unprefixed lowercase 64-hex strings.
        "parent_checkpoint_digest": _nullable_schema("string"),
    }
)
_MARKET_TAIL_ROW_SCHEMA = _object_schema(
    {
        "date": "iso_date",
        "aapl_open": "number",
        "aapl_close": "number",
        "aapl_adj_close": "number",
        "aapl_adj_open": "number",
        "spy_adj_close": "number",
        "qqq_adj_close": "number",
    }
)
_COOLDOWN_TAIL_ROW_SCHEMA = _object_schema(
    {"date": "iso_date", **{name: "bool" for name in _COOLDOWN_SIGNAL_KEYS}}
)
_REPLAY_CHECKPOINT_SCHEMA = _object_schema(
    {
        "schema_version": "int",
        "checkpoint_date": "iso_date",
        "source_start_date": "iso_date",
        "source_session_count": "int",
        "runtime_segments": _list_schema(_RUNTIME_SEGMENT_SCHEMA),
        "fixed_feature_columns": "list[string]",
        "forecast_columns": "list[string]",
        "matured_event_columns": "list[string]",
        "model_payload": _MODEL_PAYLOAD_SCHEMA,
        "pending_lessons": _list_schema(_PENDING_LESSON_SCHEMA),
        "market_feature_tail": _list_schema(_MARKET_TAIL_ROW_SCHEMA),
        "signal_cooldown_tail": _list_schema(_COOLDOWN_TAIL_ROW_SCHEMA),
        "cooldown_predecessor": _fixed_map_schema(
            _COOLDOWN_PREDECESSOR_KEYS, "bool"
        ),
        "opportunity_counts": _fixed_map_schema(_OPPORTUNITY_KEYS, "int"),
        "opportunity_hashes": _fixed_map_schema(_OPPORTUNITY_KEYS, "string"),
        "market_prefix_sha256": "string",
        "fixed_feature_prefix_sha256": "string",
        "forecast_prefix_sha256": "string",
        "matured_event_prefix_sha256": "string",
        "model_state_sha256": "string",
        "pending_state_sha256": "string",
        "market_history_tail_sha256": "string",
        "signal_cooldown_tail_sha256": "string",
        "digest_sha256": "string",
    }
)
_ACCOUNT_STATE_SCHEMA = _object_schema(
    {
        "schema_version": "int",
        "policy_name": "string",
        "cost_bps": "float",
        "inception_fill_date": _nullable_schema("iso_date"),
        "inception_count": "int",
        "last_session_date": _nullable_schema("iso_date"),
        "last_reference_price": _nullable_schema("float"),
        "last_trade_fill_date": _nullable_schema("iso_date"),
        "cash": "float",
        "shares": "float",
        "held_target": _nullable_schema("int"),
        "previous_requested_target": _nullable_schema("int"),
        "pending_decision_date": _nullable_schema("iso_date"),
        "pending_target_exposure": _nullable_schema("int"),
        "last_equity": "float",
        "running_peak": "float",
        "cumulative_active_log_edge": "float",
        "ledger_row_count": "int",
        "ledger_tip_sha256": "sha256",
        "open_cash_entry_decision_date": _nullable_schema("iso_date"),
        "open_cash_entry_fill_date": _nullable_schema("iso_date"),
        "open_cash_entry_reference_price": _nullable_schema("float"),
        "open_cash_fill_observations": "int",
    }
)
_ACCOUNT_CHECKPOINT_SCHEMA = _object_schema(
    {
        "checkpoint_schema_version": "int",
        "account_state": _ACCOUNT_STATE_SCHEMA,
        "account_state_sha256": "sha256",
    }
)
_PARENT_ACCOUNT_IDENTITY_SCHEMA = _object_schema(
    {
        "account_state_sha256": "sha256",
        "ledger_tip_sha256": "sha256",
        "ledger_row_count": "int",
    }
)
_PARENT_PAYLOAD_NAMES = tuple(
    sorted(
        {
            ".gitattributes",
            "AUDIT_ATTEMPT_LOCK.json",
            "rejected_parent_manifest.json",
            "rejected_parent_checksums.json",
            "rejected_parent_verification.json",
            "input_provenance.json",
            "audit_prices_through_2023.csv",
            "audit_fixed_features.table.json",
            "audit_forecast__fixed_comparators.table.json",
            "audit_state_weight_diagnostics.table.json",
            "audit_replay_diagnostics.json",
            "audit_pending_lessons.json",
            "audit_continuation_checkpoint_through_2023.json",
            "audit_metrics.json",
            "audit_integrity_evidence.json",
            "audit_gate_report.json",
            "audit_parent_causal_prefix_proof.json",
            "audit_prefix_continuity_proof.json",
            "source_bundle_provenance.json",
            "audit_runtime_cost_evidence.json",
            "report.json",
            *(
                f"audit_forecast__{arm}.table.json"
                for arm in _base_artifacts.ARM_ORDER
            ),
            *(
                f"audit_matured_lessons__{arm}.table.json"
                for arm in _base_artifacts.ARM_ORDER
            ),
            *(
                f"audit_ledger__{cost}__{policy}.table.json"
                for cost in _base_artifacts.COST_ORDER
                for policy in _base_artifacts.POLICY_ORDER
            ),
            *(
                f"audit_episodes__{cost}.json"
                for cost in _base_artifacts.COST_ORDER
            ),
            *(
                f"audit_xor__{cost}.json"
                for cost in _base_artifacts.COST_ORDER
            ),
        }
    )
)
if len(_PARENT_PAYLOAD_NAMES) != 51:
    raise RuntimeError("the frozen parent payload schema inventory changed")

_PARENT_VERIFICATION_SCHEMA = _object_schema(
    {
        "verifier_id": "string",
        "verifier_dependency_path": "string",
        "verified": "bool",
        "parent_directory": "string",
        "preservation_commit": "git_commit",
        "original_run_commit": "git_commit",
        "contract_version": "string",
        "stage": "string",
        "run_id": "string",
        "status": "string",
        "stage_pass": "bool",
        "historical_policy_candidate": "bool",
        "learning_candidate": "bool",
        "parent_rejection_remains_final": "bool",
        "post_2023_market_values_accessed": "bool",
        "passed_gate_count": "int",
        "total_gate_count": "int",
        "manifest_file_sha256": "sha256",
        "manifest_self_sha256": "sha256",
        "checksums_file_sha256": "sha256",
        "report_file_sha256": "sha256",
        "report_self_sha256": "sha256",
        "gate_report_file_sha256": "sha256",
        "metrics_file_sha256": "sha256",
        "integrity_file_sha256": "sha256",
        "integrity_self_sha256": "sha256",
        "checkpoint_file_sha256": "sha256",
        "checkpoint_self_sha256": "sha256",
        "exact_payload_names": "list[string]",
        "payload_sha256": _fixed_map_schema(_PARENT_PAYLOAD_NAMES, "sha256"),
        "selected_checkpoint": _object_schema(
            {
                "policy": "string",
                "digest_sha256": "string",
                "checkpoint_date": "iso_date",
                "source_session_count": "int",
                "runtime": _RUNTIME_CONTROL_SCHEMA,
                "processed_session_count": "int",
                "admitted_lesson_count": "int",
                "matured_lesson_count": "int",
                "pending_lesson_count": "int",
                "model_state_sha256": "string",
                "pending_state_sha256": "string",
            }
        ),
        "account_identity": _fixed_map_schema(
            COST_ORDER,
            _fixed_map_schema(
                (
                    "online_full",
                    "aapl_buy_hold",
                    "always_long",
                    "exact_union_cash",
                    "contextual_only",
                ),
                _PARENT_ACCOUNT_IDENTITY_SCHEMA,
            ),
        ),
    }
)

_PREFIX_CONTINUITY_SCHEMA = _object_schema(
    {
        "prefix_proof_schema_version": _literal_schema(1),
        "through_2023_rows": "int",
        "through_2023_last_session": "iso_date",
        "through_2023_canonical_sha256": "sha256",
        # The inherited replay checkpoint digest is unprefixed 64-hex.
        "parent_checkpoint_sha256": "string",
        "parent_checkpoint_source_sessions": "int",
        "prefix_market_chain_exact": "bool",
        "suffix_first_session": "iso_date",
        "suffix_last_session": "iso_date",
        "suffix_rows": "int",
        "suffix_released_only_after_prefix_proof": "bool",
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "evidence_classification": _literal_schema(
            "canonical_prefix_and_checkpoint_continuity"
        ),
        "proof_sha256": "sha256",
    },
    self_hash_field="proof_sha256",
)

_SEALED_PREPARATION_LINEAGE_SCHEMA = _object_schema(
    {
        "quarantined_record_raw_sha256": "sha256",
        "quarantined_record_git_blob": "string",
        "source_raw_sha256": "sha256",
        "source_git_blob": "string",
        "source_bundle_manifest_file_sha256": "sha256",
        "source_bundle_manifest_self_sha256": "sha256",
        "source_bundle_checksums_file_sha256": "sha256",
        "source_bundle_input_provenance_file_sha256": "sha256",
    }
)
_SOURCE_PROVENANCE_SCHEMA = _object_schema(
    {
        "provenance_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "evidence_classification": _literal_schema("authorized_source_lineage"),
        "parent": _object_schema(
            {
                "directory": "string",
                "manifest_file_sha256": "sha256",
                "manifest_self_sha256": "sha256",
                "checksums_file_sha256": "sha256",
                "checkpoint_file_sha256": "sha256",
                "checkpoint_self_sha256": "sha256",
                "selected_checkpoint_sha256": "string",
            }
        ),
        "bounded_input": _object_schema(
            {
                "path": "string",
                "raw_sha256": "sha256",
                "canonical_sha256": "sha256",
                "date_sequence_sha256": "string",
                "first_session": "iso_date",
                "last_session": "iso_date",
                "rows": "int",
                "physical_snapshot_has_later_rows": "bool",
                "rows_after_bound_returned": "bool",
            }
        ),
        "sanitized_receipt": _object_schema(
            {
                "path": "string",
                "raw_sha256": "sha256",
                "classification": "string",
                "sealed_preparation_lineage": _SEALED_PREPARATION_LINEAGE_SCHEMA,
            }
        ),
        "attempt_lock": _object_schema(
            {
                "path": "string",
                "raw_sha256": "sha256",
                "created_before_market_release": "bool",
            }
        ),
        "access_boundary": _object_schema(
            {
                "receipt_read_after_lock": "bool",
                "input_read_after_receipt_and_lock": "bool",
                "suffix_released_after_prefix_proof": "bool",
                "later_market_data_accessed": "bool",
                "prior_2024_artifact_accessed": "bool",
            }
        ),
        "provenance_sha256": "sha256",
    },
    self_hash_field="provenance_sha256",
)

_KNOWN_RESULT_ISOLATION_SCHEMA = _object_schema(
    {
        "isolation_evidence_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "evidence_classification": _literal_schema("known_result_access_firewall"),
        "dependency_paths": "list[string]",
        "dependency_paths_sha256": "sha256",
        "forbidden_path_fragments": "list[string]",
        "dependency_inventory_excludes_prior_2024_results": "bool",
        "prior_2024_forecasts_accessed": "bool",
        "prior_2024_actions_accessed": "bool",
        "prior_2024_ledgers_accessed": "bool",
        "prior_2024_metrics_accessed": "bool",
        "prior_2024_gates_accessed": "bool",
        "later_market_data_accessed": "bool",
        "expected_2024_result_fixture_used": "bool",
        "reporting_module_imported_by_policy_replay": "bool",
        "isolation_evidence_sha256": "sha256",
    },
    self_hash_field="isolation_evidence_sha256",
)

_AUDIT_CHECKPOINT_SCHEMA = _object_schema(
    {
        "checkpoint_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "checkpoint_cutoff": "iso_date",
        "source_start_session": "iso_date",
        "source_session_count": "int",
        "scenario_order": _list_schema("string", length=len(SCENARIO_ORDER)),
        "cost_order": _list_schema("string", length=len(COST_ORDER)),
        "policy_order": _list_schema("string", length=len(POLICY_ORDER)),
        "parent_identity": _object_schema(
            {
                "manifest_file_sha256": "sha256",
                "manifest_self_sha256": "sha256",
                "checkpoint_file_sha256": "sha256",
                "checkpoint_self_sha256": "sha256",
                "selected_checkpoint_sha256": "string",
                "through_2023_canonical_sha256": "sha256",
            }
        ),
        "replay_checkpoints": _fixed_map_schema(
            SCENARIO_ORDER, _REPLAY_CHECKPOINT_SCHEMA
        ),
        "administrative_accounts": _fixed_map_schema(
            COST_ORDER, _fixed_map_schema(POLICY_ORDER, _ACCOUNT_CHECKPOINT_SCHEMA)
        ),
        "parent_administrative_account_identity": _fixed_map_schema(
            COST_ORDER,
            _fixed_map_schema(POLICY_ORDER, _PARENT_ACCOUNT_IDENTITY_SCHEMA),
        ),
        "checkpoint_sha256": "sha256",
    },
    self_hash_field="checkpoint_sha256",
)

_PENDING_ARTIFACT_SCHEMA = _object_schema(
    {
        "pending_artifact_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "stage": _literal_schema(AUDIT_STAGE),
        "scenario_order": _list_schema("string", length=len(SCENARIO_ORDER)),
        "by_scenario": _fixed_map_schema(
            SCENARIO_ORDER,
            _object_schema(
                {
                    "pending_lessons": _list_schema(_PENDING_LESSON_SCHEMA),
                    "pending_lesson_count": "int",
                    "pending_state_sha256": "string",
                    "signal_cooldown_tail": _list_schema(
                        _COOLDOWN_TAIL_ROW_SCHEMA
                    ),
                    "signal_cooldown_tail_sha256": "string",
                    "cooldown_predecessor": _fixed_map_schema(
                        _COOLDOWN_PREDECESSOR_KEYS, "bool"
                    ),
                }
            ),
        ),
        "artifact_sha256": "sha256",
    },
    self_hash_field="artifact_sha256",
)

_REPLAY_DIAGNOSTIC_COUNTS_SCHEMA = _object_schema(
    {
        "processed_rows": "int",
        "accepted_opportunities": "int",
        "cash_actions": "int",
        "matured_events": "int",
        "admitted_events": "int",
        "ending_pending_lessons": "int",
        "fixed_feature_sha256": "string",
        "forecast_sha256": "string",
        "checkpoint_sha256": "string",
    }
)
_REPLAY_DIAGNOSTICS_SCHEMA = _object_schema(
    {
        "replay_diagnostics_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "stage": _literal_schema(AUDIT_STAGE),
        "scenario_order": _list_schema("string", length=len(SCENARIO_ORDER)),
        "parent_checkpoint_sha256": "string",
        "dedicated_two_scenario_fork": "bool",
        "inherited_four_arm_helper_used": "bool",
        "common_fixed_features": "bool",
        "by_scenario": _fixed_map_schema(
            SCENARIO_ORDER,
            _object_schema(
                {
                    "scenario_name": "string",
                    "initial_checkpoint_sha256": "string",
                    "initial_model_state_sha256": "string",
                    "initial_pending_state_sha256": "string",
                    "runtime": _RUNTIME_CONTROL_SCHEMA,
                    "diagnostics": _REPLAY_DIAGNOSTIC_COUNTS_SCHEMA,
                    "terminal_checkpoint_sha256": "string",
                    "terminal_model_state_sha256": "string",
                    "terminal_pending_state_sha256": "string",
                    "action_stream_sha256": "sha256",
                }
            ),
        ),
        "frozen_post_cutoff_admitted_events": "int",
        "frozen_terminal_admitted_count_equals_parent": "bool",
        "shadow_causal_maturity_order_inherited": "bool",
        "replay_diagnostics_sha256": "sha256",
    },
    self_hash_field="replay_diagnostics_sha256",
)

_COMPLETE_EPISODE_RECORD_SCHEMA = _object_schema(
    {
        "episode_id": "sha256",
        "entry_decision_date": "iso_date",
        "entry_fill_date": "iso_date",
        "exit_decision_date": "iso_date",
        "exit_fill_date": "iso_date",
        "entry_reference_price": "float",
        "entry_sell_fill_price": "float",
        "exit_reference_price": "float",
        "exit_buy_fill_price": "float",
        "cash_fill_observations": "int",
        "raw_active_log_edge": "float",
        "cost_log_edge": "float",
        "net_active_log_edge": "float",
        "episode_sha256": "sha256",
    }
)
_OPEN_EPISODE_RECORD_SCHEMA = _object_schema(
    {
        "status": _enum_schema(
            "open_cash_pending_exit",
            "open_cash",
            "pending_cash_entry_unexecuted",
        ),
        "entry_decision_date": _nullable_schema("iso_date"),
        # An unexecuted pending entry uses the frozen empty-string sentinel.
        "entry_fill_date": "string",
        "entry_reference_price": "float",
        "pending_decision_date": _nullable_schema("iso_date"),
        "pending_target_exposure": _nullable_schema("int"),
        "mark_date": "iso_date",
        "mark_reference_price": "float",
        "raw_active_log_edge_to_mark": "float",
        "executed_cost_log_edge": "float",
        "net_active_log_edge_to_mark": "float",
        "unresolved_sha256": "sha256",
    }
)
_CASH_RECONCILIATION_SCHEMA = _object_schema(
    {
        "ledger_active_log_edge": "float",
        "complete_episode_active_log_edge": "float",
        "open_episode_active_log_edge": "float",
        "reconciliation_error": "float",
        "tolerance": "float",
        "terminal_episode_force_closed": "bool",
        "passed": "bool",
    }
)
_EPISODE_ARTIFACT_SCHEMA = _object_schema(
    {
        "episode_artifact_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "stage": _literal_schema(AUDIT_STAGE),
        "cost": _enum_schema(*COST_ORDER),
        "cost_bps": "float",
        "policy_order": _list_schema("string", length=len(POLICY_ORDER)),
        "policies": _fixed_map_schema(
            POLICY_ORDER,
            _object_schema(
                {
                    "complete_columns": "list[string]",
                    "complete": _list_schema(_COMPLETE_EPISODE_RECORD_SCHEMA),
                    "open_columns": "list[string]",
                    "open": _list_schema(_OPEN_EPISODE_RECORD_SCHEMA),
                    "reconciliation": _CASH_RECONCILIATION_SCHEMA,
                }
            ),
        ),
        "artifact_sha256": "sha256",
    },
    self_hash_field="artifact_sha256",
)

_XOR_ORIENTATIONS = (
    "shadow_cash_lead_long",
    "shadow_long_lead_cash",
)
_XOR_COMPONENT_RECORD_SCHEMA = _object_schema(
    {
        "fill_date": "iso_date",
        "reference_adjusted_open": "float",
        "shadow_decision_date": "iso_date",
        "lead_decision_date": "iso_date",
        "shadow_before": "int",
        "shadow_after": "int",
        "lead_before": "int",
        "lead_after": "int",
        "orientation_after": _enum_schema(*_XOR_ORIENTATIONS, "equal"),
        "raw_market_component": "float",
        "transition_cost_component": "float",
        "net_incremental_log_edge": "float",
        "component_sha256": "sha256",
    }
)
_COMPLETE_XOR_RECORD_SCHEMA = _object_schema(
    {
        "xor_id": "sha256",
        "entry_fill_date": "iso_date",
        "exit_fill_date": "iso_date",
        "entry_quarter": "string",
        "start_shadow_decision_date": "iso_date",
        "start_lead_decision_date": "iso_date",
        "end_shadow_decision_date": "iso_date",
        "end_lead_decision_date": "iso_date",
        "orientation": _enum_schema(*_XOR_ORIENTATIONS),
        "xor_fill_observations": "int",
        "raw_market_component": "float",
        "transition_cost_component": "float",
        "net_incremental_log_edge": "float",
        "xor_sha256": "sha256",
    }
)
_OPEN_XOR_RECORD_SCHEMA = _object_schema(
    {
        "status": _literal_schema("right_boundary_partial"),
        "entry_fill_date": "iso_date",
        "last_fill_date": "iso_date",
        "entry_quarter": "string",
        "start_shadow_decision_date": "iso_date",
        "start_lead_decision_date": "iso_date",
        "last_shadow_decision_date": "iso_date",
        "last_lead_decision_date": "iso_date",
        "orientation": _enum_schema(*_XOR_ORIENTATIONS),
        "xor_fill_observations": "int",
        "raw_market_component": "float",
        "transition_cost_component": "float",
        "net_incremental_log_edge": "float",
        "open_xor_sha256": "sha256",
    }
)
_XOR_RECONCILIATION_SCHEMA = _object_schema(
    {
        "full_period_incremental_active_log_edge": "float",
        "component_incremental_active_log_edge": "float",
        "complete_xor_incremental_active_log_edge": "float",
        "open_xor_incremental_active_log_edge": "float",
        "component_reconciliation_error": "float",
        "episode_reconciliation_error": "float",
        "economic_xor_fill_count": "int",
        "complete_xor_count": "int",
        "open_xor_count": "int",
        "terminal_xor_force_closed": "bool",
        "tolerance": "float",
        "passed": "bool",
    }
)
_XOR_ARTIFACT_SCHEMA = _object_schema(
    {
        "xor_artifact_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "stage": _literal_schema(AUDIT_STAGE),
        "cost": _enum_schema(*COST_ORDER),
        "cost_bps": "float",
        "orientation_order": _list_schema(
            "string", length=len(_XOR_ORIENTATIONS)
        ),
        "component_columns": "list[string]",
        "components": _list_schema(_XOR_COMPONENT_RECORD_SCHEMA),
        "complete_columns": "list[string]",
        "complete": _list_schema(_COMPLETE_XOR_RECORD_SCHEMA),
        "open_columns": "list[string]",
        "open": _list_schema(_OPEN_XOR_RECORD_SCHEMA),
        "reconciliation": _XOR_RECONCILIATION_SCHEMA,
        "artifact_sha256": "sha256",
    },
    self_hash_field="artifact_sha256",
)

_QUARTER_KEYS = ("Q1", "Q2", "Q3", "Q4")
_LEARNING_STATUS_VALUES = (
    "unexercised",
    "exercised_insufficient_evidence",
    "exercised_positive",
    "exercised_negative",
    "exercised_flat",
)
_STAGE_STATUS_VALUES = (
    "POST_HOC_FROZEN_POLICY_2024_REPLICATION_PASS",
    "AAPL_BEATEN_BUT_MODEL_NOT_SELECTED",
    "REJECTED_2024_INTEGRITY",
    "REJECTED_2024",
)


def _criterion_schema(check_fields: Mapping[str, Any]) -> dict[str, Any]:
    return _object_schema(
        {
            "passed": "bool",
            "passed_count": "int",
            "applicable_count": "int",
            "total_count": "int",
            "checks": _object_schema(check_fields),
            "failed_checks": "list[string]",
            "not_applicable_checks": "list[string]",
        }
    )


_ACCOUNT_PERIOD_METRICS_SCHEMA = _object_schema(
    {
        "start_equity": "float",
        "end_equity": "float",
        "log_return": "float",
        "simple_return": "float",
        "fill_count": "int",
        "trade_count": "int",
        "target_change_count": "int",
        "turnover_reference": "float",
        "maximum_drawdown": "float",
    }
)
_RELATIVE_PERIOD_SCHEMA = _object_schema(
    {
        "strategy": _ACCOUNT_PERIOD_METRICS_SCHEMA,
        "aapl_buy_hold": _ACCOUNT_PERIOD_METRICS_SCHEMA,
        "active_log_edge": "float",
        "relative_wealth_advantage": "float",
    }
)


def _statistics_schema(*, year_keys: tuple[str, ...] | None = None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "count": "int",
        "positive_count": "int",
        "positive_rate": "float",
        "mean": _nullable_schema("float"),
        "median": _nullable_schema("float"),
        "positive_concentration": _nullable_schema("float"),
        "sum": "float",
    }
    if year_keys is not None:
        fields["year_edges"] = _fixed_map_schema(year_keys, "float")
    return _object_schema(fields)


_OPEN_EPISODE_DIAGNOSTICS_SCHEMA = _object_schema(
    {
        "count": "int",
        "statuses": "list[string]",
        "net_active_log_edge_to_mark": "float",
    }
)
_CALENDAR_EPISODE_STATISTICS_SCHEMA = _statistics_schema(year_keys=("2024",))
_CONTINUOUS_EPISODE_STATISTICS_SCHEMA = _statistics_schema(
    year_keys=tuple(str(year) for year in range(2005, 2025))
)
_POLICY_REPORT_SCHEMA = _object_schema(
    {
        "scenario_name": "string",
        "internal_policy_name": "string",
        "calendar_2024": _object_schema(
            {
                "full_year": _RELATIVE_PERIOD_SCHEMA,
                "quarters": _fixed_map_schema(
                    _QUARTER_KEYS, _RELATIVE_PERIOD_SCHEMA
                ),
                "complete_episode_statistics": _CALENDAR_EPISODE_STATISTICS_SCHEMA,
                "open_episode_diagnostics": _OPEN_EPISODE_DIAGNOSTICS_SCHEMA,
            }
        ),
        "continuous_2005_2024": _object_schema(
            {
                **_RELATIVE_PERIOD_SCHEMA["fields"],
                "complete_episode_statistics": _CONTINUOUS_EPISODE_STATISTICS_SCHEMA,
                "open_episode_diagnostics": _OPEN_EPISODE_DIAGNOSTICS_SCHEMA,
                "complete_plus_open_episode_edge": "float",
            }
        ),
    }
)
_XOR_METRIC_REPORT_SCHEMA = _object_schema(
    {
        "incremental_active_log_edge": "float",
        "relative_wealth_advantage": "float",
        "component_incremental_active_log_edge": "float",
        "complete_xor_incremental_active_log_edge": "float",
        "open_xor_incremental_active_log_edge": "float",
        "reconciliation_error": "float",
        "economic_xor_fill_count": "int",
        "complete_xor_episode_count": "int",
        "open_xor_episode_count": "int",
        "distinct_xor_entry_quarters": "list[string]",
        "edge_after_removing_best_complete_xor_episode": _nullable_schema(
            "float"
        ),
        "complete_xor_statistics": _statistics_schema(),
        "complete_orientations": "list[string]",
        "open_orientations": "list[string]",
        "complete_xor_episode_diagnostics": _list_schema(
            _COMPLETE_XOR_RECORD_SCHEMA
        ),
        "open_xor_episode_diagnostics": _list_schema(_OPEN_XOR_RECORD_SCHEMA),
        "evidence_reconciliation": _XOR_RECONCILIATION_SCHEMA,
    }
)
_LEARNING_BY_COST_SCHEMA = _object_schema(
    {
        "economic_xor_fill_count": "int",
        "complete_xor_episode_count": "int",
        "distinct_xor_entry_quarters": "list[string]",
        "incremental_active_log_edge": "float",
        "edge_after_removing_best_complete_xor_episode": _nullable_schema(
            "float"
        ),
    }
)
_LEARNING_CRITERION_SCHEMA = _criterion_schema(
    {
        "complete_xor_episodes_at_least_3_both_costs": "bool",
        "xor_entries_in_at_least_2_quarters_both_costs": "bool",
        "incremental_active_log_edge_gt_0_001_both_costs": "bool",
        "stress_edge_after_removing_best_complete_xor_positive": "bool",
    }
)
_LEARNING_METRICS_SCHEMA = _object_schema(
    {
        "classification": _enum_schema(*_LEARNING_STATUS_VALUES),
        "learning_candidate_for_2025_shadow": "bool",
        "candidate_criterion": _LEARNING_CRITERION_SCHEMA,
        "by_cost": _fixed_map_schema(COST_ORDER, _LEARNING_BY_COST_SCHEMA),
    }
)
_METRICS_SCHEMA = _object_schema(
    {
        "policy_order": _list_schema("string", length=len(POLICY_ORDER)),
        "policy_by_cost": _fixed_map_schema(
            COST_ORDER, _fixed_map_schema(POLICY_ORDER, _POLICY_REPORT_SCHEMA)
        ),
        "online_shadow_minus_frozen_lead_by_cost": _fixed_map_schema(
            COST_ORDER, _XOR_METRIC_REPORT_SCHEMA
        ),
        "learning": _LEARNING_METRICS_SCHEMA,
    }
)
_METRICS_ARTIFACT_SCHEMA = _object_schema(
    {
        "metrics_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "evidence_classification": _literal_schema(
            "post_hoc_frozen_policy_2024_replication"
        ),
        "metrics": _METRICS_SCHEMA,
        "metrics_sha256": "sha256",
    },
    self_hash_field="metrics_sha256",
)

_AAPL_GATE_CRITERION_SCHEMA = _criterion_schema(
    {
        "full_year_active_log_edge_gt_0_001": "bool",
        "positive_quarters_at_least_2_of_4": "bool",
        "edge_after_removing_best_quarter_positive": "bool",
        "negative_aapl_quarter_edge_positive_if_applicable": _nullable_schema(
            "bool"
        ),
    }
)
_AAPL_GATE_SCHEMA = _object_schema(
    {
        **_AAPL_GATE_CRITERION_SCHEMA["fields"],
        "full_year_active_log_edge": "float",
        "quarter_active_log_edges": _fixed_map_schema(_QUARTER_KEYS, "float"),
        "positive_quarters": "list[string]",
        "positive_quarter_count": "int",
        "edge_after_removing_best_quarter": "float",
        "aapl_quarter_log_returns": _fixed_map_schema(_QUARTER_KEYS, "float"),
        "negative_aapl_quarters": "list[string]",
        "negative_aapl_quarter_edge_sum": "float",
        "negative_aapl_quarter_support_status": _enum_schema(
            "observed_positive",
            "observed_nonpositive",
            "not_applicable_no_negative_aapl_quarters",
        ),
    }
)
_SIMPLE_COMPARATOR_KEYS = ("contextual_only", "exact_union_cash")
_SIMPLE_SUPERIORITY_SCHEMA = _object_schema(
    {
        **_criterion_schema(
            {
                "lead_minus_contextual_only_gt_0_001": "bool",
                "lead_minus_exact_union_cash_gt_0_001": "bool",
            }
        )["fields"],
        "cost_name": _literal_schema("stress_10bps"),
        "lead_full_year_log_return": "float",
        "comparator_full_year_log_returns": _fixed_map_schema(
            _SIMPLE_COMPARATOR_KEYS, "float"
        ),
        "lead_minus_comparator_log_returns": _fixed_map_schema(
            _SIMPLE_COMPARATOR_KEYS, "float"
        ),
    }
)

_STATIC_INTEGRITY_CHECK_KEYS = (
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
_COMPUTATION_INTEGRITY_BASE_KEYS = (
    "exact_parent_model_and_account_fork",
    "no_account_reset_or_added_capital",
    "frozen_lead_zero_post_cutoff_admissions",
    "online_shadow_causal_runtime",
    "binary_forecast_targets",
    "fixed_features_equal_between_scenarios",
    "forecast_frozen_before_ledger_interpretation",
)
_COMPUTATION_INTEGRITY_DYNAMIC_KEYS = tuple(
    [
        f"{cost}__{policy}__{suffix}"
        for cost in COST_ORDER
        for policy in POLICY_ORDER
        for suffix in ("no_leverage", "cash_episode_reconciliation")
    ]
    + [f"{cost}__always_long_aapl_economic_equality" for cost in COST_ORDER]
    + [f"{policy}__cross_cost_action_identity" for policy in POLICY_ORDER]
    + [f"{cost}__fill_based_xor_reconciliation" for cost in COST_ORDER]
)
_INTEGRITY_CHECK_KEYS = tuple(
    sorted(
        set(_STATIC_INTEGRITY_CHECK_KEYS)
        | set(_COMPUTATION_INTEGRITY_BASE_KEYS)
        | set(_COMPUTATION_INTEGRITY_DYNAMIC_KEYS)
    )
)
_INTEGRITY_CHECK_MAP_SCHEMA = _fixed_map_schema(_INTEGRITY_CHECK_KEYS, "bool")
_FATAL_INTEGRITY_GATE_SCHEMA = _object_schema(
    {
        "passed": "bool",
        "passed_count": "int",
        "total_count": "int",
        "checks": _INTEGRITY_CHECK_MAP_SCHEMA,
        "failed_checks": "list[string]",
    }
)
_GATE_REPORT_SCHEMA = _object_schema(
    {
        "edge_materiality": "float",
        "reconciliation_tolerance": "float",
        "zero_classification_tolerance": "float",
        "aapl_policy_by_cost": _fixed_map_schema(COST_ORDER, _AAPL_GATE_SCHEMA),
        "simple_policy_superiority": _SIMPLE_SUPERIORITY_SCHEMA,
        "learning_classification": _enum_schema(*_LEARNING_STATUS_VALUES),
        "learning_candidate_for_2025_shadow": "bool",
        "aapl_policy_pass_by_cost": _fixed_map_schema(COST_ORDER, "bool"),
        "aapl_policy_pass": "bool",
        "simple_policy_superiority_pass": "bool",
        "fatal_integrity": _FATAL_INTEGRITY_GATE_SCHEMA,
        "stage_pass": "bool",
        "status": _enum_schema(*_STAGE_STATUS_VALUES),
    }
)
_GATE_ARTIFACT_SCHEMA = _object_schema(
    {
        "gate_report_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "evidence_classification": _literal_schema(
            "post_hoc_frozen_policy_2024_replication"
        ),
        "gate_report": _GATE_REPORT_SCHEMA,
        "gate_report_sha256": "sha256",
    },
    self_hash_field="gate_report_sha256",
)

_ZERO_USE_SCHEMA = _object_schema(
    {
        "network_access": "bool",
        "news_access": "bool",
        "llm_calls": "int",
        "api_calls": "int",
        "external_model_calls": "int",
        "external_cost_usd": "float",
    }
)
_INTEGRITY_ARTIFACT_SCHEMA = _object_schema(
    {
        "integrity_evidence_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "evidence_classification": _literal_schema("fatal_integrity_evidence"),
        "checks": _INTEGRITY_CHECK_MAP_SCHEMA,
        "references": _object_schema(
            {
                "attempt_lock_file_sha256": "sha256",
                "parent_manifest_file_sha256": "sha256",
                "parent_checkpoint_file_sha256": "sha256",
                "input_raw_sha256": "sha256",
                "input_canonical_sha256": "sha256",
                "parent_checkpoint_sha256": "string",
                "frozen_action_stream_sha256": "sha256",
                "shadow_action_stream_sha256": "sha256",
            }
        ),
        "zero_use": _ZERO_USE_SCHEMA,
        "integrity_evidence_sha256": "sha256",
    },
    self_hash_field="integrity_evidence_sha256",
)

_RUNTIME_EVIDENCE_SCHEMA = _object_schema(
    {
        "runtime_evidence_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "clock_name": _literal_schema("time.monotonic"),
        "command_start_monotonic": "float",
        "stage_deadline_seconds": "float",
        "finalization_reserve_seconds": "float",
        "samples": _object_schema(
            {
                "preseal": "float",
                "post_private_verify": _nullable_schema("float"),
                "prepromotion": _nullable_schema("float"),
            }
        ),
        "stage_internal_deadline_pass": "bool",
        "final_commit_sample_location": "string",
        **_ZERO_USE_SCHEMA["fields"],
        "runtime_evidence_sha256": "sha256",
    },
    self_hash_field="runtime_evidence_sha256",
)
_REPORT_SCHEMA = _object_schema(
    {
        "report_schema_version": _literal_schema(1),
        "contract_version": _literal_schema(CONTRACT_VERSION),
        "run_id": _literal_schema(AUDIT_RUN_ID),
        "stage": _literal_schema(AUDIT_STAGE),
        "status": _enum_schema(*_STAGE_STATUS_VALUES),
        "stage_pass": "bool",
        "evidence_classification": _literal_schema(
            "post_hoc_frozen_policy_2024_replication"
        ),
        "is_confirmation": "bool",
        "is_prospective": "bool",
        "is_unseen_2024_test": "bool",
        "post_hoc_frozen_policy_replication": "bool",
        "historical_results_authorize_real_capital": "bool",
        "calendar_period": _object_schema(
            {"first": "iso_date", "last": "iso_date"}
        ),
        "continuous_account_period": _object_schema(
            {"first_year": "int", "last_year": "int"}
        ),
        "parent_rejection_remains_final": "bool",
        "later_market_data_accessed": "bool",
        "prior_2024_artifact_accessed": "bool",
        "learning_classification": _enum_schema(*_LEARNING_STATUS_VALUES),
        "learning_candidate_for_2025_shadow": "bool",
        "metrics": _METRICS_SCHEMA,
        "gate_report": _GATE_REPORT_SCHEMA,
        "runtime_cost_evidence": _RUNTIME_EVIDENCE_SCHEMA,
        "artifact_references": _object_schema(
            {
                "checkpoint_sha256": "sha256",
                "provenance_sha256": "sha256",
                "isolation_evidence_sha256": "sha256",
                "integrity_evidence_sha256": "sha256",
            }
        ),
        "next_step": "string",
        "report_sha256": "sha256",
    },
    self_hash_field="report_sha256",
)

_JSON_ARTIFACT_SCHEMAS: Mapping[str, Any] = MappingProxyType(
    {
        "parent_verification.json": _PARENT_VERIFICATION_SCHEMA,
        "prefix_continuity_proof.json": _PREFIX_CONTINUITY_SCHEMA,
        "source_bundle_provenance.json": _SOURCE_PROVENANCE_SCHEMA,
        "known_result_isolation_evidence.json": _KNOWN_RESULT_ISOLATION_SCHEMA,
        "audit_continuation_checkpoint_through_2024.json": _AUDIT_CHECKPOINT_SCHEMA,
        "audit_pending_lessons.json": _PENDING_ARTIFACT_SCHEMA,
        "audit_replay_diagnostics.json": _REPLAY_DIAGNOSTICS_SCHEMA,
        "audit_episodes__base_5bps.json": _EPISODE_ARTIFACT_SCHEMA,
        "audit_episodes__stress_10bps.json": _EPISODE_ARTIFACT_SCHEMA,
        "audit_xor__base_5bps.json": _XOR_ARTIFACT_SCHEMA,
        "audit_xor__stress_10bps.json": _XOR_ARTIFACT_SCHEMA,
        "audit_metrics.json": _METRICS_ARTIFACT_SCHEMA,
        "audit_gate_report.json": _GATE_ARTIFACT_SCHEMA,
        "audit_integrity_evidence.json": _INTEGRITY_ARTIFACT_SCHEMA,
        "audit_runtime_cost_evidence.json": _RUNTIME_EVIDENCE_SCHEMA,
        "report.json": _REPORT_SCHEMA,
    }
)
if set(_JSON_ARTIFACT_SCHEMAS) != set(_JSON_ARTIFACT_ROLES):
    raise RuntimeError("the frozen JSON schema registry inventory is incomplete")

ARTIFACT_SCHEMA_REGISTRY: Mapping[str, Any] = MappingProxyType(
    {
        "registry_schema_version": 1,
        "bundle_file_order": list(BUNDLE_FILE_ORDER),
        "payload_file_order": list(PAYLOAD_FILE_ORDER),
        "checksum_file_order": list(CHECKSUM_FILE_ORDER),
        "canonical_json": {
            "encoding": "utf-8",
            "ensure_ascii": True,
            "finite_numbers_only": True,
            "object_keys_sorted": True,
            "separators": [",", ":"],
            "terminal_lf_count": 1,
            "unknown_or_missing_keys_fatal": True,
        },
        "table_artifacts": {
            filename: _schema_descriptor(
                TABLE_SCHEMA_BY_FILENAME[filename],
                rows=TABLE_EXPECTED_ROWS_BY_FILENAME[filename],
            )
            for filename in sorted(TABLE_SCHEMA_BY_FILENAME)
        },
        "json_artifacts": {
            filename: {
                "kind": "canonical_json",
                "schema_version": 1,
                "evidence_classification": classification,
                "literal_nested_schema": _JSON_ARTIFACT_SCHEMAS[filename],
                "unknown_or_missing_keys_fatal": True,
            }
            for filename, classification in sorted(_JSON_ARTIFACT_ROLES.items())
        },
        "stage_manifest": {
            "kind": "omitted_field_self_hashed_json",
            "self_hash_field": "manifest_sha256",
            "fields": sorted(_MANIFEST_FIELDS),
            "payload_keys": list(PAYLOAD_FILE_ORDER),
        },
        "checksums": {
            "kind": "raw_file_sha256_map",
            "fields": list(CHECKSUM_FILE_ORDER),
            "excluded_field": "checksums.json",
        },
        "success_marker": {
            "kind": "omitted_field_self_hashed_json",
            "outside_bundle": True,
            "self_hash_field": "marker_sha256",
            "fields": sorted(_MARKER_FIELDS),
            "inventory_hash_domain": "canonical_sorted_43_filename_json_array",
        },
    }
)
ARTIFACT_SCHEMA_REGISTRY_SHA256 = _sha256_json(dict(ARTIFACT_SCHEMA_REGISTRY))


def artifact_schema_registry_sha256() -> str:
    return _sha256_json(dict(ARTIFACT_SCHEMA_REGISTRY))


def validate_payload_inventory(payloads: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(payloads, Mapping) or set(payloads) != PAYLOAD_FILENAMES:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "payload inventory differs from the frozen 41-file domain"
        )
    result: dict[str, bytes] = {}
    for filename in PAYLOAD_FILE_ORDER:
        payload = payloads[filename]
        if type(payload) is not bytes or Path(filename).name != filename:
            raise ContextualExpertAggregation2024AuditArtifactError(
                "payload filename or bytes are invalid"
            )
        result[filename] = payload
    if result[".gitattributes"] != GIT_ATTRIBUTES_BYTES:
        raise ContextualExpertAggregation2024AuditArtifactError(
            ".gitattributes bytes differ from the frozen contract"
        )
    return result


def payload_sha256(payloads: Mapping[str, bytes]) -> dict[str, str]:
    clean = validate_payload_inventory(payloads)
    return {filename: _sha256_bytes(clean[filename]) for filename in PAYLOAD_FILE_ORDER}


_STAGE_STATUSES = frozenset(
    {
        "POST_HOC_FROZEN_POLICY_2024_REPLICATION_PASS",
        "AAPL_BEATEN_BUT_MODEL_NOT_SELECTED",
        "REJECTED_2024_INTEGRITY",
        "REJECTED_2024",
    }
)
_LEARNING_CLASSIFICATIONS = frozenset(
    {
        "unexercised",
        "exercised_insufficient_evidence",
        "exercised_positive",
        "exercised_negative",
        "exercised_flat",
    }
)


def _validate_hash_map(
    value: Any, expected_names: tuple[str, ...], *, field: str
) -> dict[str, str]:
    mapping = _strict_mapping(value, frozenset(expected_names), field=field)
    return {
        name: _require_sha256(mapping[name], field=f"{field}.{name}")
        for name in expected_names
    }


def build_stage_manifest(
    manifest_fields: Mapping[str, Any], *, payload_hashes: Mapping[str, str]
) -> dict[str, Any]:
    fields = _strict_mapping(
        manifest_fields, _MANIFEST_UNSIGNED_FIELDS - {"payload_sha256"}, field="manifest fields"
    )
    hashes = _validate_hash_map(payload_hashes, PAYLOAD_FILE_ORDER, field="payload_sha256")
    unsigned = {**dict(fields), "payload_sha256": hashes}
    _validate_manifest_unsigned(unsigned)
    return {**unsigned, "manifest_sha256": _sha256_json(unsigned)}


def _validate_manifest_unsigned(value: Mapping[str, Any]) -> None:
    _strict_mapping(value, _MANIFEST_UNSIGNED_FIELDS, field="unsigned manifest")
    if (
        value["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION
        or type(value["manifest_schema_version"]) is bool
        or value["contract_version"] != CONTRACT_VERSION
        or value["verifier_id"] != VERIFIER_ID
        or value["run_id"] != AUDIT_RUN_ID
        or value["stage"] != AUDIT_STAGE
        or value["preregistration_commit"] != PREREGISTRATION_COMMIT
        or value["artifact_schema_registry_sha256"]
        != ARTIFACT_SCHEMA_REGISTRY_SHA256
        or value["parent_manifest_file_sha256"] != PARENT_MANIFEST_FILE_SHA256
        or value["parent_manifest_self_sha256"] != PARENT_MANIFEST_SELF_SHA256
        or value["parent_checkpoint_file_sha256"] != PARENT_CHECKPOINT_FILE_SHA256
        or value["parent_checkpoint_self_sha256"] != PARENT_CHECKPOINT_SELF_SHA256
        or value["status"] not in _STAGE_STATUSES
        or type(value["stage_pass"]) is not bool
        or type(value["evidence_classification"]) is not str
        or not value["evidence_classification"]
        or not isinstance(value["git_identity"], Mapping)
        or value["learning_classification"] not in _LEARNING_CLASSIFICATIONS
        or type(value["later_market_data_accessed"]) is not bool
        or type(value["prior_2024_artifact_accessed"]) is not bool
        or type(value["external_cost_usd"]) is not float
        or value["external_cost_usd"] != 0.0
    ):
        raise ContextualExpertAggregation2024AuditArtifactError(
            "manifest frozen identity or primitive types changed"
        )
    if value["stage_pass"] != (
        value["status"] == "POST_HOC_FROZEN_POLICY_2024_REPLICATION_PASS"
    ):
        raise ContextualExpertAggregation2024AuditArtifactError(
            "manifest status and stage_pass disagree"
        )
    for name in (
        "dependency_identity_sha256",
        "artifact_schema_registry_sha256",
        "parent_manifest_file_sha256",
        "parent_manifest_self_sha256",
        "parent_checkpoint_file_sha256",
        "parent_checkpoint_self_sha256",
        "receipt_file_sha256",
        "input_raw_sha256",
        "input_canonical_sha256",
        "attempt_lock_file_sha256",
        "runtime_cost_evidence_file_sha256",
        "gate_report_file_sha256",
    ):
        _require_sha256(value[name], field=name)
    _validate_hash_map(value["payload_sha256"], PAYLOAD_FILE_ORDER, field="payload_sha256")


def parse_stage_manifest(value: Any) -> dict[str, Any]:
    manifest = _strict_mapping(value, _MANIFEST_FIELDS, field="stage manifest")
    unsigned = {name: manifest[name] for name in manifest if name != "manifest_sha256"}
    _validate_manifest_unsigned(unsigned)
    recorded = _require_sha256(manifest["manifest_sha256"], field="manifest_sha256")
    if recorded != _sha256_json(unsigned):
        raise ContextualExpertAggregation2024AuditArtifactError(
            "stage manifest omitted-field self-hash is invalid"
        )
    return dict(manifest)


def stage_manifest_bytes(value: Any) -> bytes:
    return canonical_json_line_bytes(parse_stage_manifest(value))


def parse_stage_manifest_bytes(payload: bytes) -> dict[str, Any]:
    value = parse_canonical_json_line(payload, field="stage manifest")
    return parse_stage_manifest(value)


def build_checksums(
    *, payload_hashes: Mapping[str, str], stage_manifest_payload: bytes
) -> dict[str, str]:
    hashes = _validate_hash_map(payload_hashes, PAYLOAD_FILE_ORDER, field="payload_sha256")
    if type(stage_manifest_payload) is not bytes:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "stage manifest payload must be exact bytes"
        )
    parse_stage_manifest_bytes(stage_manifest_payload)
    return {**hashes, "stage_manifest.json": _sha256_bytes(stage_manifest_payload)}


def parse_checksums(value: Any) -> dict[str, str]:
    return _validate_hash_map(value, CHECKSUM_FILE_ORDER, field="checksums")


def checksums_bytes(value: Any) -> bytes:
    return canonical_json_line_bytes(parse_checksums(value))


def parse_checksums_bytes(payload: bytes) -> dict[str, str]:
    value = parse_canonical_json_line(payload, field="checksums")
    return parse_checksums(value)


def final_inventory_sha256() -> str:
    return _sha256_json(sorted(BUNDLE_FILE_ORDER))


FINAL_INVENTORY_SHA256 = final_inventory_sha256()


def _validate_marker_unsigned(value: Mapping[str, Any]) -> None:
    _strict_mapping(value, _MARKER_UNSIGNED_FIELDS, field="unsigned success marker")
    elapsed = value["marker_preparation_elapsed_seconds"]
    deadline = value["stage_deadline_seconds"]
    if (
        value["marker_schema_version"] != MARKER_SCHEMA_VERSION
        or type(value["marker_schema_version"]) is bool
        or value["contract_version"] != CONTRACT_VERSION
        or value["run_id"] != AUDIT_RUN_ID
        or value["stage"] != AUDIT_STAGE
        or value["preregistration_commit"] != PREREGISTRATION_COMMIT
        or value["final_relative_path"] != OUTPUT_DIRECTORY.as_posix()
        or value["final_inventory_sha256"] != FINAL_INVENTORY_SHA256
        or type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed < 0.0
        or type(deadline) is not float
        or deadline != STAGE_DEADLINE_SECONDS
        or type(value["marker_preparation_deadline_pass"]) is not bool
        or value["marker_preparation_deadline_pass"] is not True
        or not elapsed < deadline
        or type(value["external_cost_usd"]) is not float
        or value["external_cost_usd"] != 0.0
    ):
        raise ContextualExpertAggregation2024AuditArtifactError(
            "success marker frozen identity, strict timing, or primitive types changed"
        )
    _require_commit(value["git_commit"], field="git_commit")
    for name in (
        "attempt_lock_file_sha256",
        "final_inventory_sha256",
        "stage_manifest_file_sha256",
        "stage_manifest_self_sha256",
        "checksums_file_sha256",
    ):
        _require_sha256(value[name], field=name)


def build_success_marker(marker_fields: Mapping[str, Any]) -> dict[str, Any]:
    fields = _strict_mapping(marker_fields, _MARKER_UNSIGNED_FIELDS, field="marker fields")
    unsigned = dict(fields)
    _validate_marker_unsigned(unsigned)
    return {**unsigned, "marker_sha256": _sha256_json(unsigned)}


def parse_success_marker(value: Any) -> dict[str, Any]:
    marker = _strict_mapping(value, _MARKER_FIELDS, field="success marker")
    unsigned = {name: marker[name] for name in marker if name != "marker_sha256"}
    _validate_marker_unsigned(unsigned)
    recorded = _require_sha256(marker["marker_sha256"], field="marker_sha256")
    if recorded != _sha256_json(unsigned):
        raise ContextualExpertAggregation2024AuditArtifactError(
            "success marker omitted-field self-hash is invalid"
        )
    return dict(marker)


def success_marker_bytes(value: Any) -> bytes:
    return canonical_json_line_bytes(parse_success_marker(value))


def parse_success_marker_bytes(payload: bytes) -> dict[str, Any]:
    value = parse_canonical_json_line(payload, field="success marker")
    return parse_success_marker(value)


@dataclass(frozen=True)
class BundleMetadata:
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    checksums: Mapping[str, str]
    checksums_bytes: bytes


def build_bundle_metadata(
    payloads: Mapping[str, bytes], *, manifest_fields: Mapping[str, Any]
) -> BundleMetadata:
    hashes = payload_sha256(payloads)
    manifest = build_stage_manifest(manifest_fields, payload_hashes=hashes)
    manifest_payload = stage_manifest_bytes(manifest)
    sums = build_checksums(
        payload_hashes=hashes, stage_manifest_payload=manifest_payload
    )
    sums_payload = checksums_bytes(sums)
    return BundleMetadata(
        manifest=manifest,
        manifest_bytes=manifest_payload,
        checksums=sums,
        checksums_bytes=sums_payload,
    )


def write_private_bundle(
    directory: Path,
    *,
    payloads: Mapping[str, bytes],
    metadata: BundleMetadata,
) -> None:
    """Exclusively write and fsync one private 43-file bundle, without promotion."""

    clean = validate_payload_inventory(payloads)
    if not isinstance(metadata, BundleMetadata):
        raise ContextualExpertAggregation2024AuditArtifactError(
            "bundle metadata has the wrong type"
        )
    expected = build_bundle_metadata(
        clean,
        manifest_fields={
            key: metadata.manifest[key]
            for key in _MANIFEST_UNSIGNED_FIELDS
            if key != "payload_sha256"
        },
    )
    if metadata != expected:
        raise ContextualExpertAggregation2024AuditArtifactError(
            "bundle metadata differs from deterministic reconstruction"
        )
    target = Path(directory)
    if target.exists():
        raise ContextualExpertAggregation2024AuditArtifactError(
            "private bundle directory already exists"
        )
    target.mkdir(parents=False)
    try:
        for filename in PAYLOAD_FILE_ORDER:
            _experiment._exclusive_write(target / filename, clean[filename])
        _experiment._exclusive_write(
            target / "checksums.json", metadata.checksums_bytes
        )
        _experiment._exclusive_write(
            target / "stage_manifest.json", metadata.manifest_bytes
        )
        if {entry.name for entry in target.iterdir()} != BUNDLE_FILENAMES:
            raise ContextualExpertAggregation2024AuditArtifactError(
                "private bundle inventory changed while writing"
            )
        _experiment._fsync_directory(target)
    except Exception:
        # The runner owns the frozen failure transition.  Preserve partial
        # evidence here instead of silently deleting it.
        raise


__all__ = [
    "ARTIFACT_SCHEMA_REGISTRY",
    "ARTIFACT_SCHEMA_REGISTRY_SHA256",
    "ATTEMPT_LOCK_FILENAME",
    "ATTEMPT_LOCK_PATH",
    "AUDIT_RUN_ID",
    "AUDIT_STAGE",
    "BUNDLE_FILENAMES",
    "BUNDLE_FILE_ORDER",
    "BundleMetadata",
    "CHECKSUM_FILENAMES",
    "CHECKSUM_FILE_ORDER",
    "CONTRACT_VERSION",
    "CONTROL_ROOT",
    "ContextualExpertAggregation2024AuditArtifactError",
    "DIAGNOSTIC_SCHEMA",
    "EXPECTED_BRANCH",
    "FAILED_DIRECTORY",
    "FINAL_INVENTORY_SHA256",
    "FIXED_COMPARATOR_SCHEMA",
    "FIXED_FEATURE_SCHEMA",
    "FORECAST_SCHEMA",
    "GIT_ATTRIBUTES_BYTES",
    "GIT_ATTRIBUTES_SHA256",
    "LEDGER_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "MARKER_SCHEMA_VERSION",
    "MATURED_EVENT_SCHEMA",
    "OUTPUT_DIRECTORY",
    "OUTPUT_PARENT",
    "PARENT_CHECKPOINT_FILE_SHA256",
    "PARENT_CHECKPOINT_SELF_SHA256",
    "PARENT_MANIFEST_FILE_SHA256",
    "PARENT_MANIFEST_SELF_SHA256",
    "PAYLOAD_FILENAMES",
    "PAYLOAD_FILE_ORDER",
    "PENDING_SUCCESS_MARKER_FILENAME",
    "PENDING_SUCCESS_MARKER_PATH",
    "POLICY_ORDER",
    "PREREGISTRATION_COMMIT",
    "PRIVATE_DIRECTORY",
    "SCENARIO_ORDER",
    "STAGE_DEADLINE_SECONDS",
    "SUCCESS_MARKER_FILENAME",
    "SUCCESS_MARKER_PATH",
    "TABLE_COLUMN_SHA256",
    "TABLE_EXPECTED_ROWS_BY_FILENAME",
    "TABLE_SCHEMA_BY_FILENAME",
    "VERIFIER_ID",
    "artifact_schema_registry_sha256",
    "build_bundle_metadata",
    "build_checksums",
    "build_stage_manifest",
    "build_success_marker",
    "canonical_json_line_bytes",
    "canonical_registered_table_bytes",
    "canonical_table_bytes",
    "checksums_bytes",
    "column_list_sha256",
    "final_inventory_sha256",
    "parse_canonical_json_line",
    "parse_canonical_table_bytes",
    "parse_checksums",
    "parse_checksums_bytes",
    "parse_registered_table_bytes",
    "parse_stage_manifest",
    "parse_stage_manifest_bytes",
    "parse_success_marker",
    "parse_success_marker_bytes",
    "payload_sha256",
    "stage_manifest_bytes",
    "success_marker_bytes",
    "table_schema_for_filename",
    "validate_payload_inventory",
    "write_private_bundle",
]
