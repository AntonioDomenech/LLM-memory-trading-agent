"""Strict bounded-input and sanitized-receipt verification for the 2024 audit.

Nothing in this module opens a repository path at import time.  File-reading
entry points are intended to be called only after the irreversible attempt
lock and exact-path Git checks have succeeded.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_replay as _replay
from .contextual_expert_aggregation_2024_audit_replay import (
    FROZEN_CUTOFF,
    _require_exact_parent_prefix,
)


CONTRACT_VERSION = "aapl-causal-contextual-expert-aggregation-2024-audit-v2"
INPUT_PATH = Path(
    "e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/"
    "authorized_inputs/aapl_spy_qqq_through_2024.csv"
)
RECEIPT_PATH = Path(
    "e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/"
    "input_snapshot_receipt.json"
)
INPUT_RAW_SHA256 = (
    "sha256:abf8115e61e7a7724ed816db7dbd0fe053b098123b2e1e43886eb0074b575be7"
)
INPUT_GIT_BLOB = "3cdb68aac6a3b7b1cda4f64ee81193690f28a1d7"
INPUT_CANONICAL_SHA256 = (
    "sha256:5b3df584b4ccedb6f6871e6cbd43095126378485aec4f3a55f4846d2fb74f071"
)
INPUT_DATE_SEQUENCE_SHA256 = (
    "6a3a247806d34797029831a945c15458e15eabc2bd36b021d2a815668c7d5e4c"
)
RECEIPT_RAW_SHA256 = (
    "sha256:b2f541541f132f6c22257c18b36ec2f785260b105d44e35a8f0da0c35ac5fae9"
)
RECEIPT_GIT_BLOB = "cd69e5d15c3ed122c0082ae5d7f87bc84c2311f7"
PREFIX_CANONICAL_SHA256 = (
    "sha256:3b5e02acaa69a56fa47a0fd34275472d226b62c0f61741c13d3680b239b82535"
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

INPUT_SPEC = _experiment.AuthorizedPriceSpec(
    stage="audit_2024",
    relative_path=INPUT_PATH,
    raw_sha256=INPUT_RAW_SHA256,
    git_blob=INPUT_GIT_BLOB,
    first_session="1999-03-10",
    last_session="2024-12-31",
    rows=6496,
    date_sequence_sha256=INPUT_DATE_SEQUENCE_SHA256,
    canonical_sha256=INPUT_CANONICAL_SHA256,
)

RECEIPT_EXPECTED: Mapping[str, Any] = {
    "schema_version": "aapl-contextual-aggregation-through-2024-audit-receipt-v1",
    "created_for_contract": CONTRACT_VERSION,
    "classification": "sanitized_audit_input_receipt",
    "destination": {
        "path": INPUT_PATH.as_posix(),
        "schema": list(_experiment.PHYSICAL_PRICE_COLUMNS),
        "raw_sha256": INPUT_RAW_SHA256,
        "git_blob": INPUT_GIT_BLOB,
        "canonical_sha256": INPUT_CANONICAL_SHA256,
        "date_sequence_sha256": INPUT_DATE_SEQUENCE_SHA256,
        "first_session": "1999-03-10",
        "last_session": "2024-12-31",
        "rows": 6496,
        "bytes": 681685,
        "physical_snapshot_has_later_rows": False,
        "rows_after_bound_returned": False,
    },
    "prefix_authority": {
        "through_2023_rows": 6244,
        "through_2023_last_session": FROZEN_CUTOFF,
        "through_2023_canonical_sha256": PREFIX_CANONICAL_SHA256,
        "through_2023_prefix_match": True,
        "parent_audit_manifest_self_sha256": PARENT_MANIFEST_SELF_SHA256,
        "parent_checkpoint_payload_sha256": PARENT_CHECKPOINT_FILE_SHA256,
        "parent_checkpoint_self_sha256": PARENT_CHECKPOINT_SELF_SHA256,
    },
    "sealed_preparation_lineage": {
        "quarantined_record_raw_sha256": "sha256:9e013cfff01ec582b0db7a21387937c4b6a64794505ce6550ee1de91c79a5933",
        "quarantined_record_git_blob": "0c06a4dcec8e39776e2bd1cc3af931256d7b5ff7",
        "source_raw_sha256": "sha256:cf384e8218f97d4c6e7ca89860877ef5edb67e938af66886bdf81912bd35479b",
        "source_git_blob": "358ff80f066ae3ba36ee66613fc90d62dbc31a2f",
        "source_bundle_manifest_file_sha256": "sha256:b98e250b5940170bb88aa7d61b76e4ce938b27efe245fb7aa26ea54aef1fc0e1",
        "source_bundle_manifest_self_sha256": "sha256:ad59b77b27ead2012c8ae73e422ba85b26bb58121fbefedaecfa5640c7f87cdb",
        "source_bundle_checksums_file_sha256": "sha256:9c8ef690335be5304ee66f4e28cf7d262952bc4ff8af19dc819a943bbfeb9e49",
        "source_bundle_input_provenance_file_sha256": "sha256:341f6b3894544a6da332b2a5983988ae720d554dc0a2d984de7a63e3707c97e9",
    },
    "preparation_attestations": {
        "cutoff": "2024-12-31",
        "lineage_strength": "self_attested_mechanical_cut_plus_independent_destination_integrity_review",
        "standalone_cutter_pre_registered": False,
        "standalone_cutter_identity_available": False,
        "deterministic_prefix_copy": True,
        "source_identity_verified_before_copy": True,
        "destination_written_atomically": True,
        "destination_overwrite_allowed": False,
        "policy_or_evaluation_code_executed": False,
        "strategy_parameters_changed": False,
        "market_values_printed": False,
        "independent_bounded_data_integrity_review_passed": True,
        "independent_through_2023_prefix_and_checkpoint_review_passed": True,
        "network_access": False,
        "news_access": False,
        "llm_calls": 0,
        "api_calls": 0,
        "external_cost_usd": 0.0,
    },
}


class AuditInputError(RuntimeError):
    """Raised when the sanitized receipt or bounded input changes."""


@dataclass(frozen=True)
class BoundedAuditSnapshot:
    snapshot: _experiment.LoadedPriceSnapshot
    input_bytes: bytes
    receipt: Mapping[str, Any]
    receipt_bytes: bytes
    historical_prefix: pd.DataFrame
    suffix: pd.DataFrame
    prefix_continuity: Mapping[str, Any]


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _strict_same(observed: Any, expected: Any, *, field: str) -> None:
    if type(observed) is not type(expected):
        raise AuditInputError(f"sanitized receipt type changed at {field}")
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            raise AuditInputError(f"sanitized receipt keys changed at {field}")
        for key in expected:
            _strict_same(observed[key], expected[key], field=f"{field}.{key}")
        return
    if isinstance(expected, list):
        if len(observed) != len(expected):
            raise AuditInputError(f"sanitized receipt length changed at {field}")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _strict_same(left, right, field=f"{field}[{index}]")
        return
    if observed != expected:
        raise AuditInputError(f"sanitized receipt value changed at {field}")


def parse_sanitized_receipt_bytes(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or _sha256(payload) != RECEIPT_RAW_SHA256:
        raise AuditInputError("sanitized receipt raw identity changed")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditInputError("sanitized receipt is not strict UTF-8 JSON") from exc
    _strict_same(value, RECEIPT_EXPECTED, field="receipt")
    return copy.deepcopy(value)


def verify_prefix_continuity(
    snapshot: _experiment.LoadedPriceSnapshot,
    checkpoint: _replay.ReplayCheckpoint,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prove the exact through-2023 prefix before releasing the suffix."""

    if not isinstance(snapshot, _experiment.LoadedPriceSnapshot):
        raise AuditInputError("bounded input loader evidence has the wrong type")
    frame = snapshot.frame
    prefix = frame.loc[frame.index <= pd.Timestamp(FROZEN_CUTOFF)].copy()
    if (
        len(prefix) != 6244
        or prefix.index[-1].date().isoformat() != FROZEN_CUTOFF
        or _experiment.sha256_bytes(_experiment.canonical_price_csv_bytes(prefix))
        != PREFIX_CANONICAL_SHA256
    ):
        raise AuditInputError("bounded input through-2023 prefix changed")
    try:
        canonical_prefix = _require_exact_parent_prefix(prefix, checkpoint)
    except (TypeError, ValueError) as exc:
        raise AuditInputError("bounded input prefix disagrees with parent checkpoint") from exc

    # Suffix access is deliberately deferred until every prefix check above.
    suffix = frame.loc[frame.index > pd.Timestamp(FROZEN_CUTOFF)].copy()
    if (
        len(suffix) != 252
        or suffix.empty
        or suffix.index[0].year != 2024
        or suffix.index[-1].date().isoformat() != "2024-12-31"
        or any(value.year != 2024 for value in suffix.index)
    ):
        raise AuditInputError("bounded input audit suffix changed")
    proof = {
        "prefix_proof_schema_version": 1,
        "through_2023_rows": len(canonical_prefix),
        "through_2023_last_session": FROZEN_CUTOFF,
        "through_2023_canonical_sha256": PREFIX_CANONICAL_SHA256,
        "parent_checkpoint_sha256": checkpoint.digest_sha256,
        "parent_checkpoint_source_sessions": checkpoint.source_session_count,
        "prefix_market_chain_exact": True,
        "suffix_first_session": suffix.index[0].date().isoformat(),
        "suffix_last_session": suffix.index[-1].date().isoformat(),
        "suffix_rows": len(suffix),
        "suffix_released_only_after_prefix_proof": True,
    }
    return canonical_prefix, suffix, proof


def load_postlock_bounded_snapshot(
    *,
    input_path: Path,
    receipt_path: Path,
    checkpoint: _replay.ReplayCheckpoint,
    verified_receipt_bytes: bytes | None = None,
    verified_input_bytes: bytes | None = None,
) -> BoundedAuditSnapshot:
    """First authorized literal reads of receipt and input, in that order."""

    if verified_receipt_bytes is None:
        try:
            receipt_bytes = receipt_path.read_bytes()
        except OSError as exc:
            raise AuditInputError("sanitized receipt became unreadable") from exc
    else:
        receipt_bytes = verified_receipt_bytes
    receipt = parse_sanitized_receipt_bytes(receipt_bytes)
    try:
        snapshot = _experiment.load_bounded_price_snapshot(input_path, spec=INPUT_SPEC)
        input_bytes = (
            input_path.read_bytes()
            if verified_input_bytes is None
            else verified_input_bytes
        )
    except (OSError, _experiment.ContextualExpertAggregationExperimentError) as exc:
        raise AuditInputError("bounded audit input failed strict validation") from exc
    if _sha256(input_bytes) != INPUT_RAW_SHA256:
        raise AuditInputError("bounded audit input preverified bytes changed")
    if len(input_bytes) != RECEIPT_EXPECTED["destination"]["bytes"]:
        raise AuditInputError("bounded audit input byte count changed")
    prefix, suffix, proof = verify_prefix_continuity(snapshot, checkpoint)
    return BoundedAuditSnapshot(
        snapshot=snapshot,
        input_bytes=input_bytes,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        historical_prefix=prefix,
        suffix=suffix,
        prefix_continuity=proof,
    )


__all__ = [
    "AuditInputError",
    "BoundedAuditSnapshot",
    "CONTRACT_VERSION",
    "INPUT_PATH",
    "RECEIPT_PATH",
    "INPUT_RAW_SHA256",
    "INPUT_GIT_BLOB",
    "INPUT_CANONICAL_SHA256",
    "RECEIPT_RAW_SHA256",
    "RECEIPT_GIT_BLOB",
    "PREFIX_CANONICAL_SHA256",
    "INPUT_SPEC",
    "RECEIPT_EXPECTED",
    "parse_sanitized_receipt_bytes",
    "verify_prefix_continuity",
    "load_postlock_bounded_snapshot",
]
