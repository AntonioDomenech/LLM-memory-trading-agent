"""Exact identities, checkpoint envelope, and sealing for the v2 audit.

This module owns no acquisition, model decision, score, or economic gate.  It
keeps the post-rejection audit namespace separate from the frozen v2 stage
namespace while reusing the already-verified replay and ledger envelopes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from . import contextual_expert_aggregation_artifacts as _v2_artifacts
from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_ledger as _ledger
from . import contextual_expert_aggregation_replay as _replay
from .contextual_expert_aggregation_artifacts import (
    ARM_ORDER,
    COST_BPS,
    COST_ORDER,
    FIXED_POLICY_ORDER,
    POLICY_ORDER,
)


CONTRACT_VERSION = (
    "aapl-causal-contextual-expert-aggregation-2019-2023-audit-v2"
)
PARENT_CONTRACT_VERSION = "aapl-causal-contextual-expert-aggregation-v2"
AUDIT_STAGE = "audit"
AUDIT_RUN_ID = (
    "contextual-expert-aggregation-post-rejection-2019-2023-audit-v2"
)
EXPECTED_BRANCH = "codex/aapl-causal-contextual-expert-aggregation-audit-v2"
OUTPUT_PARENT = Path("e/aapl_causal_contextual_expert_aggregation_audit_v2")
OUTPUT_DIRECTORY = OUTPUT_PARENT / AUDIT_RUN_ID
ATTEMPT_LOCK_FILENAME = "AUDIT_ATTEMPT_LOCK.json"
ATTEMPT_LOCK_PATH = OUTPUT_PARENT / ATTEMPT_LOCK_FILENAME
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_CUTOFF = "2023-12-31"
LAST_OBSERVED_SESSION = "2023-12-29"
SOURCE_START_SESSION = "1999-03-10"
SOURCE_SESSION_COUNT = 6_244
GIT_ATTRIBUTES_BYTES = b"* -text\n"


class ContextualExpertAggregationAuditArtifactError(RuntimeError):
    """Raised when an audit artifact or immutable seal is malformed."""


def _sha256_json(value: Any) -> str:
    return _experiment.sha256_bytes(_experiment.canonical_json_bytes(value))


def _strict_mapping(
    value: Any, expected: set[str], *, field: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContextualExpertAggregationAuditArtifactError(
            f"{field} has the wrong exact field inventory"
        )
    return value


def payload_names() -> frozenset[str]:
    """Return the complete flat payload inventory before seal metadata."""

    names = {
        ".gitattributes",
        ATTEMPT_LOCK_FILENAME,
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
    }
    names.update(f"audit_forecast__{arm}.table.json" for arm in ARM_ORDER)
    names.update(f"audit_matured_lessons__{arm}.table.json" for arm in ARM_ORDER)
    for cost in COST_ORDER:
        names.update(
            f"audit_ledger__{cost}__{policy}.table.json"
            for policy in POLICY_ORDER
        )
        names.add(f"audit_episodes__{cost}.json")
        names.add(f"audit_xor__{cost}.json")
    return frozenset(names)


AUDIT_PAYLOAD_NAMES = payload_names()


def build_audit_checkpoint(
    *,
    replay_checkpoints: Mapping[str, _replay.ReplayCheckpoint | Mapping[str, Any]],
    administrative_accounts: Mapping[
        str, Mapping[str, _ledger.AccountState | Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    """Build and self-hash the terminal model plus account continuation state."""

    if set(replay_checkpoints) != set(ARM_ORDER):
        raise ContextualExpertAggregationAuditArtifactError(
            "audit replay checkpoint inventory changed"
        )
    normalized_replays: dict[str, _replay.ReplayCheckpoint] = {}
    for arm in ARM_ORDER:
        raw = replay_checkpoints[arm]
        checkpoint = (
            raw
            if isinstance(raw, _replay.ReplayCheckpoint)
            else _replay.ReplayCheckpoint.from_dict(raw)
        )
        checkpoint.validate()
        if (
            checkpoint.checkpoint_date != LAST_OBSERVED_SESSION
            or checkpoint.source_start_date != SOURCE_START_SESSION
            or checkpoint.source_session_count != SOURCE_SESSION_COUNT
        ):
            raise ContextualExpertAggregationAuditArtifactError(
                "audit replay checkpoint has the wrong source boundary"
            )
        normalized_replays[arm] = checkpoint

    try:
        _v2_artifacts._validate_replay_stage_semantics(
            _v2_artifacts.CONFIRMATION_STAGE, normalized_replays
        )
    except _v2_artifacts.ContextualExpertAggregationArtifactError as exc:
        raise ContextualExpertAggregationAuditArtifactError(
            "audit replay checkpoints violate exact v2 continuation semantics"
        ) from exc

    try:
        normalized_accounts = _v2_artifacts._normalize_administrative_accounts(
            administrative_accounts, cutoff=LAST_OBSERVED_SESSION
        )
        _v2_artifacts._require_always_long_benchmark_equivalence(
            normalized_accounts
        )
    except (
        _v2_artifacts.ContextualExpertAggregationArtifactError,
        _ledger.BinaryLedgerError,
    ) as exc:
        raise ContextualExpertAggregationAuditArtifactError(
            "audit account checkpoints violate exact v2 continuation semantics"
        ) from exc

    unsigned = {
        "contract_version": CONTRACT_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage": AUDIT_STAGE,
        "checkpoint_cutoff": CHECKPOINT_CUTOFF,
        "last_observed_session": LAST_OBSERVED_SESSION,
        "source_start_session": SOURCE_START_SESSION,
        "source_session_count": SOURCE_SESSION_COUNT,
        "arm_order": list(ARM_ORDER),
        "policy_order": list(POLICY_ORDER),
        "fixed_policy_order": list(FIXED_POLICY_ORDER),
        "cost_order": list(COST_ORDER),
        "replay_checkpoints": {
            arm: normalized_replays[arm].to_dict() for arm in ARM_ORDER
        },
        "administrative_accounts": {
            cost: {
                policy: normalized_accounts[cost][policy].to_checkpoint()
                for policy in POLICY_ORDER
            }
            for cost in COST_ORDER
        },
    }
    return {**unsigned, "checkpoint_sha256": _sha256_json(unsigned)}


_CHECKPOINT_FIELDS = {
    "contract_version",
    "checkpoint_schema_version",
    "stage",
    "checkpoint_cutoff",
    "last_observed_session",
    "source_start_session",
    "source_session_count",
    "arm_order",
    "policy_order",
    "fixed_policy_order",
    "cost_order",
    "replay_checkpoints",
    "administrative_accounts",
    "checkpoint_sha256",
}


def parse_audit_checkpoint(value: Any) -> dict[str, Any]:
    """Strictly parse and deterministically regenerate an audit checkpoint."""

    payload = _strict_mapping(value, _CHECKPOINT_FIELDS, field="audit checkpoint")
    if (
        payload["contract_version"] != CONTRACT_VERSION
        or payload["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or payload["stage"] != AUDIT_STAGE
        or payload["checkpoint_cutoff"] != CHECKPOINT_CUTOFF
        or payload["last_observed_session"] != LAST_OBSERVED_SESSION
        or payload["source_start_session"] != SOURCE_START_SESSION
        or payload["source_session_count"] != SOURCE_SESSION_COUNT
        or payload["arm_order"] != list(ARM_ORDER)
        or payload["policy_order"] != list(POLICY_ORDER)
        or payload["fixed_policy_order"] != list(FIXED_POLICY_ORDER)
        or payload["cost_order"] != list(COST_ORDER)
    ):
        raise ContextualExpertAggregationAuditArtifactError(
            "audit checkpoint frozen metadata changed"
        )
    expected = build_audit_checkpoint(
        replay_checkpoints=payload["replay_checkpoints"],
        administrative_accounts=payload["administrative_accounts"],
    )
    if dict(payload) != expected:
        raise ContextualExpertAggregationAuditArtifactError(
            "audit checkpoint differs from deterministic regeneration"
        )
    return expected


def audit_checkpoint_bytes(value: Any) -> bytes:
    parsed = parse_audit_checkpoint(value)
    return _experiment.canonical_json_bytes(parsed) + b"\n"


def parse_audit_checkpoint_bytes(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise ContextualExpertAggregationAuditArtifactError(
            "audit checkpoint payload must be exact bytes"
        )
    try:
        import json

        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationAuditArtifactError(
            "audit checkpoint is not strict JSON"
        ) from exc
    parsed = parse_audit_checkpoint(value)
    if payload != audit_checkpoint_bytes(parsed):
        raise ContextualExpertAggregationAuditArtifactError(
            "audit checkpoint JSON is not canonical"
        )
    return parsed


def _safe_payloads(
    payloads: Mapping[str, bytes], expected_names: set[str] | frozenset[str]
) -> dict[str, bytes]:
    if not isinstance(payloads, Mapping):
        raise ContextualExpertAggregationAuditArtifactError(
            "audit payloads must be a mapping"
        )
    result: dict[str, bytes] = {}
    for raw_name, raw_payload in payloads.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or Path(raw_name).name != raw_name
            or raw_name in {"stage_manifest.json", "checksums.json"}
            or not isinstance(raw_payload, bytes)
        ):
            raise ContextualExpertAggregationAuditArtifactError(
                "audit payload name or bytes are invalid"
            )
        result[raw_name] = raw_payload
    if set(result) != set(expected_names):
        raise ContextualExpertAggregationAuditArtifactError(
            "audit payload inventory differs from the frozen inventory"
        )
    return result


def seal_audit_bundle(
    final_directory: Path,
    *,
    manifest_fields: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    deadline: _experiment.StageDeadline,
    before_promote: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> _experiment.SealedBundle:
    """Write, independently inspect, and atomically promote one audit bundle."""

    deadline.check("audit seal entry")
    final = _experiment._lexical_absolute(final_directory)
    parent = _experiment._harden_path(
        final.parent, label="Audit seal parent", require_exists=True
    )
    # Keep the private path compact on Windows while retaining the immutable
    # run ID as the verified bundle basename.
    temporary_root = parent / ".audit-v2.sealing"
    temporary = temporary_root / AUDIT_RUN_ID
    failed = parent / ".audit-v2.failed"
    parent_identity = _experiment._directory_identity(
        parent, label="Audit seal parent"
    )
    if final.name != AUDIT_RUN_ID or final != parent / AUDIT_RUN_ID:
        raise ContextualExpertAggregationAuditArtifactError(
            "audit final directory does not match its frozen run ID"
        )
    if final.exists() or temporary_root.exists() or failed.exists():
        raise ContextualExpertAggregationAuditArtifactError(
            "audit final or stale sealing directory already exists"
        )
    entries = list(parent.iterdir())
    if (
        {entry.name for entry in entries} != {ATTEMPT_LOCK_FILENAME}
        or any(
            not entry.is_file() or _experiment._path_is_reparse_point(entry)
            for entry in entries
        )
    ):
        raise ContextualExpertAggregationAuditArtifactError(
            "audit output parent must contain only the durable attempt lock"
        )
    fields = dict(manifest_fields)
    if (
        fields.get("stage") != AUDIT_STAGE
        or fields.get("run_id") != AUDIT_RUN_ID
        or type(fields.get("stage_pass")) is not bool
        or any(
            name in fields
            for name in ("contract_version", "payload_sha256", "manifest_sha256")
        )
    ):
        raise ContextualExpertAggregationAuditArtifactError(
            "audit manifest fields violate the frozen seal identity"
        )
    clean_payloads = _safe_payloads(payloads, AUDIT_PAYLOAD_NAMES)
    hashes = {
        name: _experiment.sha256_bytes(value)
        for name, value in sorted(clean_payloads.items())
    }
    manifest = _experiment.self_hashed_manifest(
        {
            "contract_version": CONTRACT_VERSION,
            **fields,
            "payload_sha256": hashes,
        }
    )
    manifest_bytes = _experiment.pretty_json_bytes(manifest)
    checksums = {
        **hashes,
        "stage_manifest.json": _experiment.sha256_bytes(manifest_bytes),
    }
    checksums_bytes = _experiment.pretty_json_bytes(checksums)
    deadline.check("after audit seal construction")

    temporary_root.mkdir()
    temporary.mkdir()
    temporary_root_identity = _experiment._directory_identity(
        temporary_root, label="Audit temporary root"
    )
    temporary_identity = _experiment._directory_identity(
        temporary, label="Audit temporary bundle"
    )
    promoted = False
    renamed = False
    try:
        for filename, value in clean_payloads.items():
            _experiment._exclusive_write(temporary / filename, value)
        _experiment._exclusive_write(
            temporary / "stage_manifest.json", manifest_bytes
        )
        _experiment._exclusive_write(temporary / "checksums.json", checksums_bytes)
        if before_promote is not None:
            before_promote(temporary, manifest)
        if {entry.name for entry in parent.iterdir()} != {
            ATTEMPT_LOCK_FILENAME,
            temporary_root.name,
        }:
            raise ContextualExpertAggregationAuditArtifactError(
                "unexpected audit output appeared during private sealing"
            )
        _experiment._require_same_directory_identity(
            parent, parent_identity, label="Audit seal parent"
        )
        _experiment._require_same_directory_identity(
            temporary_root,
            temporary_root_identity,
            label="Audit temporary root",
        )
        if {entry.name for entry in temporary_root.iterdir()} != {AUDIT_RUN_ID}:
            raise ContextualExpertAggregationAuditArtifactError(
                "audit temporary root gained an unexpected entry"
            )
        _experiment._require_same_directory_identity(
            temporary, temporary_identity, label="Audit temporary bundle"
        )
        verified = _experiment._verify_bundle_identity(
            temporary,
            expected_contract_version=CONTRACT_VERSION,
            expected_stage=AUDIT_STAGE,
            expected_manifest_sha256=manifest["manifest_sha256"],
            expected_payload_names=AUDIT_PAYLOAD_NAMES,
            require_stage_pass=bool(fields["stage_pass"]),
        )
        if {entry.name for entry in temporary.iterdir()} != {
            *AUDIT_PAYLOAD_NAMES,
            "stage_manifest.json",
            "checksums.json",
        }:
            raise ContextualExpertAggregationAuditArtifactError(
                "audit private bundle gained an unexpected entry"
            )
        _experiment._fsync_directory(temporary)
        deadline.check("immediately before audit promotion")
        _experiment._require_same_directory_identity(
            parent, parent_identity, label="Audit seal parent"
        )
        os.replace(temporary, final)
        renamed = True
        _experiment._require_same_directory_identity(
            final, temporary_identity, label="Promoted audit bundle"
        )
        _experiment._require_same_directory_identity(
            temporary_root,
            temporary_root_identity,
            label="Audit temporary root",
        )
        if any(temporary_root.iterdir()):
            raise ContextualExpertAggregationAuditArtifactError(
                "audit temporary root is not empty after promotion"
            )
        temporary_root.rmdir()
        _experiment._fsync_directory(parent)
        promoted = True
    finally:
        if not promoted:
            _experiment._require_same_directory_identity(
                parent, parent_identity, label="Audit seal parent"
            )
            bundle_cleanup = temporary
            if renamed and final.exists():
                _experiment._require_same_directory_identity(
                    final,
                    temporary_identity,
                    label="Failed promoted audit bundle",
                )
                if failed.exists():
                    raise ContextualExpertAggregationAuditArtifactError(
                        "audit failure quarantine already exists"
                    )
                os.replace(final, failed)
                bundle_cleanup = failed
            if bundle_cleanup.exists():
                _experiment._require_same_directory_identity(
                    bundle_cleanup,
                    temporary_identity,
                    label="Failed audit private bundle",
                )
                shutil.rmtree(bundle_cleanup)
            if temporary_root.exists():
                _experiment._require_same_directory_identity(
                    temporary_root,
                    temporary_root_identity,
                    label="Failed audit temporary root",
                )
                shutil.rmtree(temporary_root)
            if final.exists() or failed.exists() or temporary_root.exists():
                raise ContextualExpertAggregationAuditArtifactError(
                    "failed audit bundle cleanup did not complete"
                )
            if renamed:
                _experiment._fsync_directory(parent)
    return _experiment.SealedBundle(
        directory=final,
        manifest_path=final / "stage_manifest.json",
        manifest=dict(verified.manifest),
        checksums=dict(verified.checksums),
    )


__all__ = [
    "ATTEMPT_LOCK_FILENAME",
    "ATTEMPT_LOCK_PATH",
    "AUDIT_PAYLOAD_NAMES",
    "AUDIT_RUN_ID",
    "AUDIT_STAGE",
    "CHECKPOINT_CUTOFF",
    "CONTRACT_VERSION",
    "ContextualExpertAggregationAuditArtifactError",
    "EXPECTED_BRANCH",
    "GIT_ATTRIBUTES_BYTES",
    "LAST_OBSERVED_SESSION",
    "OUTPUT_DIRECTORY",
    "OUTPUT_PARENT",
    "PARENT_CONTRACT_VERSION",
    "SOURCE_SESSION_COUNT",
    "SOURCE_START_SESSION",
    "audit_checkpoint_bytes",
    "build_audit_checkpoint",
    "parse_audit_checkpoint",
    "parse_audit_checkpoint_bytes",
    "payload_names",
    "seal_audit_bundle",
]
