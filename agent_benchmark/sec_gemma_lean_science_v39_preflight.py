"""Once-only, zero-external-effect preflight for v3.9 science.

The preflight authenticates the pushed implementation and the read-only v3.8
bridge, freezes aggregate request commitments, runs the offline suite, and
seals both a private content-addressed manifest and a redacted public artifact.
It never calls SEC, Yahoo, Ollama, a broker, or a paid service.

All effectful dependencies are explicit.  Tests inject synthetic public values;
the command-line adapter lazily connects the local Git, bridge, runner, and
offline-test implementations only when the user deliberately runs preflight.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from dataclasses import dataclass, replace
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from . import sec_gemma_lean_science_v39_contract as contract


PREFLIGHT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-preflight-v1"
)
PRIVATE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-private-preflight-manifest-v1"
)
PUBLIC_ARTIFACT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-public-preflight-v1"
)
PUBLIC_FAILED_PREFLIGHT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-public-preflight-failure-v1"
)
REQUEST_COMMITMENTS_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-request-commitments-v1"
)
PREFLIGHT_INTENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-preflight-intent-v1"
)
OFFLINE_TEST_REPORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-offline-tests-v1"
)
EXECUTION_AUTHORITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-execution-authority-v1"
)
PUSHED_RESULT_GATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-pushed-result-gate-v1"
)
TERMINAL_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-terminal-evidence-v1"
)
PUSHED_RESULT_PAUSE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-development-pause-v1"
)
PUSHED_RESULT_RUNTIME_GUARD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-runtime-segment-guard-v1"
)
PUSHED_RESULT_LATENCY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-latency-receipt-v1"
)
PREFLIGHT_ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-preflight-001"
)
PRIVATE_INTENT_FILENAME: Final[str] = "preflight-intent.json"
PRIVATE_MANIFEST_DIRECTORY: Final[str] = "manifests"
PRIVATE_COMPLETION_FILENAME: Final[str] = "preflight-complete.json"
PRIVATE_CONFIG_PATH: Final[str] = "data/local_config.json"
V38_PRIVATE_ROOT: Final[str] = "data/aapl_sec_gemma_lean_evidence_v3_8"
MAX_PRIVATE_CONFIG_BYTES: Final[int] = 64 * 1024
MAX_RESULT_ARTIFACT_BYTES: Final[int] = 32 * 1024 * 1024
MAX_COMPARISON_BYTES: Final[int] = 4 * 1024 * 1024
_TREE_SCAN_CHUNK_BYTES: Final[int] = 1024 * 1024
_REPARSE_ATTRIBUTE: Final[int] = 0x400
OFFLINE_TEST_TIMEOUT_SECONDS: Final[int] = 6 * 60 * 60
OFFLINE_TEST_COMMAND_PROFILE: Final[dict[str, Any]] = {
    "argv_tail": ["-m", "pytest", "-q"],
    "cwd": "repository_root",
    "stdout": "discarded",
    "stderr": "discarded",
    "timeout_seconds": OFFLINE_TEST_TIMEOUT_SECONDS,
    "network_authorized": False,
}
OFFLINE_TEST_COMMAND_PROFILE_SHA256: Final[str] = contract.canonical_sha256(
    OFFLINE_TEST_COMMAND_PROFILE
)

_SHA1_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_ACCESSION_RE: Final[re.Pattern[bytes]] = re.compile(
    rb"[0-9]{10}-[0-9]{2}-[0-9]{6}"
)

_BRIDGE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "document_count",
        "source_authority_pins_sha256",
        "source_authority_base_commit",
        "source_authority_base_tree",
        "source_inventory_sha256",
        "stage_source_seal_sha256",
        "checkpoint_sha256",
        "compact_replay_sha256",
        "role_manifests_sha256",
        "role_plan_sha256",
        "science_projection_sha256",
        "legacy_projection_sha256",
        "documents_sha256",
        "records_sha256",
        "source_order_sha256",
        "event_order_sha256",
        "prior_links_sha256",
        "primary_documents_sha256",
        "set_parity",
        "source_sequence_parity",
        "legacy_projection_parity",
        "typed_identity_parity",
        "prior_links_are_internal_only",
        "first_10k_and_10q_have_no_prior",
        "peak_live_complete_submission_blob_count",
        "sec_request_count",
        "confirmation_or_final_opened",
        "contains_private_rows",
        "contains_accessions_urls_filenames_or_bodies",
        "bridge_sha256",
    }
)
_REQUEST_COMMITMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "document_count",
        "record_count",
        "event_count",
        "request_count",
        "pilot_count",
        "remaining_count",
        "documents_sha256",
        "records_sha256",
        "events_sha256",
        "preprocessed_events_sha256",
        "canonical_requests_sha256",
        "pilot_order_sha256",
        "remaining_order_sha256",
        "source_order_sha256",
        "prior_links_sha256",
        "set_parity",
        "sequence_parity",
        "prior_link_parity",
        "confirmation_or_final_opened",
        "contains_private_rows",
    }
)
_SOURCE_INVENTORY_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"path", "git_blob_sha1", "literal_sha256", "byte_count"}
)
_REPOSITORY_SNAPSHOT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ancestry",
        "source_inventory",
        "source_inventory_sha256",
        "production_source_inventory_sha256",
        "test_source_inventory_sha256",
        "predecessor_inventory_sha256",
        "unchanged_predecessor_inventory_sha256",
        "preregistration_authenticated",
    }
)
_OFFLINE_TEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema_version", "suite", "passed", "exit_code", "command_profile_sha256"}
)
_PRIVATE_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "preflight_attempt_id",
        "status",
        "contract_manifest_sha256",
        "repository",
        "bridge",
        "request_commitments",
        "effect_counts_before",
        "effect_counts_after",
        "offline_tests",
        "privacy",
        "gates",
        "private_manifest_sha256",
    }
)
_PUBLIC_ARTIFACT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "preflight_attempt_id",
        "status",
        "stage",
        "branch",
        "preregistration_commit",
        "preregistration_tree",
        "implementation_commit",
        "implementation_tree",
        "contract_manifest_sha256",
        "science_projection_sha256",
        "source_authority",
        "counts",
        "aggregates",
        "effect_counts",
        "offline_tests",
        "privacy",
        "gates",
        "private_manifest_sha256",
        "private_manifest_literal_sha256",
        "eligible_for_public_seal",
        "development_authorized",
        "public_artifact_sha256",
    }
)
_PUBLIC_FAILED_PREFLIGHT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "preflight_attempt_id",
        "status",
        "stage",
        "branch",
        "failure_code",
        "preflight_consumed",
        "failure_preserved",
        "redacted_error_only",
        "eligible_for_public_seal",
        "rerun_authorized",
        "development_authorized",
        "confirmation_and_final_opened",
        "real_money_authorized",
        "public_artifact_sha256",
    }
)
_PUSHED_RESULT_PAUSE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "status",
        "stage",
        "attempt_id",
        "branch",
        "preflight_commit",
        "implementation_commit",
        "source_bridge_sha256",
        "science_projection_sha256",
        "model_plan_sha256",
        "pilot_count",
        "pilot_durations_ns",
        "formula",
        "projected_ns",
        "threshold_ns",
        "strictly_greater_pause",
        "pilot_guard_sha256",
        "latency_receipt_sha256",
        "remaining_order_sha256",
        "effect_report",
        "model_responses_opened",
        "market_values_opened",
        "sixth_generation_attempted",
        "continuation_requires_fresh_explicit_permission",
        "confirmation_and_final_opened",
        "privacy_passed",
        "pause_artifact_sha256",
    }
)
_PUSHED_RESULT_PILOT_GUARD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "segment_id",
        "store_segment_id",
        "generation_count",
        "pre_runtime_receipt_sha256",
        "post_runtime_receipt_sha256",
        "stable_runtime_identity_sha256",
        "ordered_generation_response_event_sha256s",
        "ordered_generation_response_events_sha256",
        "raw_show_hash_is_diagnostic_only",
        "modified_at_is_excluded_only",
        "identity_http_request_count",
        "retry_count",
        "segment_guard_sha256",
    }
)
_PUSHED_RESULT_LATENCY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "selection",
        "remaining_order",
        "pilot_rows",
        "pilot_order_sha256",
        "pilot_count",
        "remaining_count",
        "formula",
        "projected_ns",
        "threshold_ns",
        "pause_required",
        "timing_interval",
        "latency_receipt_sha256",
    }
)
_PUSHED_RESULT_LATENCY_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"execution_ordinal", "request_sha256", "request_byte_count", "duration_ns"}
)
_PRIVATE_GATES: Final[frozenset[str]] = frozenset(
    {
        "implementation_ancestry_passed",
        "exact_twelve_file_delta_passed",
        "predecessor_blobs_unchanged",
        "source_authority_authenticated",
        "streaming_projection_parity_passed",
        "request_and_pilot_commitments_frozen",
        "confirmation_and_final_unreachable",
        "zero_external_effects",
        "offline_tests_passed",
        "privacy_passed",
        "eligible_for_public_seal",
    }
)
_PUBLIC_GATES: Final[frozenset[str]] = _PRIVATE_GATES - {
    "eligible_for_public_seal"
}
_PRIVATE_PRIVACY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "aggregate_only_bridge_serialized",
        "aggregate_only_request_commitments_serialized",
        "readable_contact_stored",
        "absolute_private_path_stored",
    }
)
_PUBLIC_PRIVACY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "readable_contact_published",
        "accessions_urls_filenames_offsets_bodies_published",
        "private_paths_published",
        "canonical_requests_published",
        "pre_release_science_values_published",
        "redacted_errors_only",
    }
)


class V39PreflightError(RuntimeError):
    """A fixed public-safe preflight rejection."""

    def __init__(self, code: str) -> None:
        if type(code) is not str or re.fullmatch(r"[a-z0-9_]{3,80}", code) is None:
            code = "preflight_rejected"
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PreflightDependencies:
    """Injected read-only/pure dependencies for the once-only preflight."""

    inspect_repository: Callable[[Path], Mapping[str, Any]]
    authenticate_source: Callable[[Path], Any]
    build_projection: Callable[[Any], Any]
    build_request_commitments: Callable[[Any], Mapping[str, Any]]
    effect_snapshot: Callable[[], Mapping[str, Any]]
    run_offline_tests: Callable[[Path], Mapping[str, Any]]
    privacy_tokens: Callable[[Path], Sequence[bytes | str]]


@dataclass(frozen=True)
class PushedResultDependencies:
    """Injected read-only dependencies for the separately pushed result gate."""

    authenticate_preflight_revision: Callable[[Path, str], Mapping[str, Any]]
    load_private_contact: Callable[[Path], str]
    authenticate_source: Callable[[Path, str], Any]
    build_projection: Callable[[Any], Any]
    rebuild_attempt_context: Callable[
        [Mapping[str, Any], Any], Mapping[str, Any]
    ]
    open_store: Callable[[Path, Mapping[str, Any]], Any]
    build_effect_report: Callable[[Any], Mapping[str, Any]]
    build_public_effect_report: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    rebuild_pilot_evidence: Callable[..., tuple[Mapping[str, Any], Mapping[str, Any]]]
    build_public_pause_artifact: Callable[..., Mapping[str, Any]]
    build_continuation_preregistration: Callable[[Mapping[str, Any]], bytes]
    replay_completed_terminal: Callable[..., Mapping[str, Any]]
    replay_failure_terminal: Callable[..., Mapping[str, Any]]
    build_public_terminal_artifact: Callable[..., Mapping[str, Any]]
    build_comparison_update: Callable[[bytes, Mapping[str, Any]], bytes]
    privacy_tokens: Callable[[Path, str], Sequence[bytes | str]]


@dataclass(frozen=True)
class PublicationRecoveryDependencies:
    """Read-only builders used to finish an interrupted public publication."""

    authenticate_preflight_revision: Callable[[Path, str], Mapping[str, Any]]
    open_store: Callable[[Path, Mapping[str, Any]], Any]
    build_public_terminal_artifact: Callable[..., Mapping[str, Any]]
    build_comparison_update: Callable[[bytes, Mapping[str, Any]], bytes]
    rebuild_public_pause_artifact: Callable[..., Mapping[str, Any]] | None = None


def _reject(code: str) -> None:
    raise V39PreflightError(code)


def _is_sha1(value: Any) -> bool:
    return type(value) is str and _SHA1_RE.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _reject(code)
    return _plain_json_mapping(value, code=code)


def _plain_json_mapping(value: Any, *, code: str) -> dict[str, Any]:
    """Recursively thaw MappingProxy/tuple evidence through canonical JSON."""

    def thaw(item: Any, *, depth: int) -> Any:
        if depth > 200:
            _reject(code)
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if type(key) is not str or key in result:
                    _reject(code)
                result[key] = thaw(child, depth=depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [thaw(child, depth=depth + 1) for child in item]
        if item is None or type(item) in {str, bool, int, float}:
            return item
        _reject(code)

    detached = thaw(value, depth=0)
    if type(detached) is not dict:
        _reject(code)
    try:
        contract.canonical_json_bytes(detached)
    except Exception:
        _reject(code)
    return detached


def _artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return contract.canonical_json_bytes(dict(value)) + b"\n"


def _self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        _reject("preflight_self_hash_invalid")
    detached = copy.deepcopy(dict(value))
    detached[field] = contract.canonical_sha256(detached)
    return detached


def _validate_self_hash(value: Any, field: str, *, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _reject(code)
    detached = _plain_json_mapping(value, code=code)
    observed = detached.pop(field, None)
    if not _is_sha256(observed) or observed != contract.canonical_sha256(detached):
        _reject(code)
    return _plain_json_mapping(value, code=code)


def _validate_source_inventory(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(contract.IMPLEMENTATION_ALLOWED_PATHS):
        _reject("preflight_source_inventory_invalid")
    rows: list[dict[str, Any]] = []
    for expected_path, candidate in zip(contract.IMPLEMENTATION_ALLOWED_PATHS, value):
        row = _strict_mapping(
            candidate,
            _SOURCE_INVENTORY_ROW_FIELDS,
            code="preflight_source_inventory_invalid",
        )
        if (
            row["path"] != expected_path
            or not _is_sha1(row["git_blob_sha1"])
            or not _is_sha256(row["literal_sha256"])
            or type(row["byte_count"]) is not int
            or row["byte_count"] <= 0
        ):
            _reject("preflight_source_inventory_invalid")
        rows.append(row)
    return rows


def validate_repository_snapshot(value: Any) -> dict[str, Any]:
    """Validate exact pushed implementation ancestry and source inventory."""

    item = _strict_mapping(
        value,
        _REPOSITORY_SNAPSHOT_FIELDS,
        code="preflight_repository_shape_invalid",
    )
    try:
        ancestry = contract.validate_implementation_ancestry(item["ancestry"])
    except contract.ContractViolation:
        _reject("preflight_implementation_ancestry_invalid")
    rows = _validate_source_inventory(item["source_inventory"])
    production_rows = rows[: len(contract.IMPLEMENTATION_PRODUCTION_PATHS)]
    test_rows = rows[len(contract.IMPLEMENTATION_PRODUCTION_PATHS) :]
    expected_all = contract.canonical_sha256(rows)
    expected_production = contract.canonical_sha256(production_rows)
    expected_tests = contract.canonical_sha256(test_rows)
    if (
        item["source_inventory_sha256"] != expected_all
        or item["production_source_inventory_sha256"] != expected_production
        or item["test_source_inventory_sha256"] != expected_tests
        or not _is_sha256(item["predecessor_inventory_sha256"])
        or item["unchanged_predecessor_inventory_sha256"]
        != item["predecessor_inventory_sha256"]
        or item["preregistration_authenticated"] is not True
    ):
        _reject("preflight_repository_identity_invalid")
    return {
        **item,
        "ancestry": ancestry,
        "source_inventory": rows,
    }


def validate_bridge_manifest(value: Any) -> dict[str, Any]:
    """Accept only the aggregate, exact-development bridge receipt."""

    item = _strict_mapping(
        value,
        _BRIDGE_MANIFEST_FIELDS,
        code="preflight_bridge_manifest_shape_invalid",
    )
    body = dict(item)
    observed_hash = body.pop("bridge_sha256", None)
    hash_fields = {
        "source_authority_pins_sha256",
        "source_inventory_sha256",
        "stage_source_seal_sha256",
        "checkpoint_sha256",
        "compact_replay_sha256",
        "role_manifests_sha256",
        "role_plan_sha256",
        "science_projection_sha256",
        "legacy_projection_sha256",
        "documents_sha256",
        "records_sha256",
        "source_order_sha256",
        "event_order_sha256",
        "prior_links_sha256",
        "primary_documents_sha256",
    }
    if any(not _is_sha256(item[field]) for field in hash_fields):
        _reject("preflight_bridge_manifest_hash_invalid")
    if (
        not _is_sha256(observed_hash)
        or observed_hash != contract.canonical_sha256(body)
        or item["stage"] != contract.DEVELOPMENT_COMMAND
        or item["document_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["source_authority_pins_sha256"]
        != contract.canonical_sha256(contract.build_source_authority_pins())
        or item["source_authority_base_commit"] != contract.BASE_COMMIT
        or item["source_authority_base_tree"] != contract.BASE_TREE
        or item["source_inventory_sha256"] != contract.V38_INVENTORY_SHA256
        or item["stage_source_seal_sha256"] != contract.V38_STAGE_SOURCE_SEAL_SHA256
        or item["checkpoint_sha256"] != contract.V38_LOGICAL_CHECKPOINT_SHA256
        or item["compact_replay_sha256"] != contract.V38_COMPACT_REPLAY_SHA256
        or item["role_manifests_sha256"]
        != contract.V38_ROLE_MANIFEST_INVENTORY_SHA256
        or item["role_plan_sha256"] != contract.V38_ROLE_PLAN_SHA256
        or item["science_projection_sha256"] != contract.SCIENCE_PROJECTION_SHA256
        or item["set_parity"] is not True
        or item["source_sequence_parity"] is not True
        or item["legacy_projection_parity"] is not True
        or item["typed_identity_parity"] is not True
        or item["prior_links_are_internal_only"] is not True
        or item["first_10k_and_10q_have_no_prior"] is not True
        or item["peak_live_complete_submission_blob_count"] != 1
        or item["sec_request_count"] != 0
        or item["confirmation_or_final_opened"] is not False
        or item["contains_private_rows"] is not False
        or item["contains_accessions_urls_filenames_or_bodies"] is not False
    ):
        _reject("preflight_bridge_manifest_invalid")
    return item


def validate_request_commitments(value: Any) -> dict[str, Any]:
    """Validate the exact aggregate-only 75-request/pilot commitment."""

    item = _strict_mapping(
        value,
        _REQUEST_COMMITMENT_FIELDS,
        code="preflight_request_commitments_shape_invalid",
    )
    hash_fields = {
        "documents_sha256",
        "records_sha256",
        "events_sha256",
        "preprocessed_events_sha256",
        "canonical_requests_sha256",
        "pilot_order_sha256",
        "remaining_order_sha256",
        "source_order_sha256",
        "prior_links_sha256",
    }
    if any(not _is_sha256(item[field]) for field in hash_fields):
        _reject("preflight_request_commitments_hash_invalid")
    if (
        item["schema_version"] != REQUEST_COMMITMENTS_SCHEMA_VERSION
        or item["stage"] != contract.DEVELOPMENT_COMMAND
        or item["document_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["record_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["event_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["request_count"] != contract.DEVELOPMENT_DOCUMENT_COUNT
        or item["pilot_count"] != contract.DEVELOPMENT_PILOT_COUNT
        or item["remaining_count"] != contract.DEVELOPMENT_REMAINING_COUNT
        or item["set_parity"] is not True
        or item["sequence_parity"] is not True
        or item["prior_link_parity"] is not True
        or item["confirmation_or_final_opened"] is not False
        or item["contains_private_rows"] is not False
    ):
        _reject("preflight_request_commitments_invalid")
    return item


def _validate_bridge_commitment_parity(
    bridge: Mapping[str, Any],
    commitments: Mapping[str, Any],
) -> None:
    if (
        commitments["documents_sha256"] != bridge["documents_sha256"]
        or commitments["source_order_sha256"] != bridge["source_order_sha256"]
        or commitments["prior_links_sha256"] != bridge["prior_links_sha256"]
    ):
        _reject("preflight_bridge_request_parity_invalid")


def validate_offline_test_report(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _OFFLINE_TEST_FIELDS,
        code="preflight_offline_tests_shape_invalid",
    )
    if (
        item["schema_version"] != OFFLINE_TEST_REPORT_SCHEMA_VERSION
        or item["suite"] != "complete_repository_pytest"
        or item["passed"] is not True
        or item["exit_code"] != 0
        or item["command_profile_sha256"] != OFFLINE_TEST_COMMAND_PROFILE_SHA256
    ):
        _reject("preflight_offline_tests_failed")
    return item


def _validate_effect_snapshot(value: Any) -> dict[str, int]:
    try:
        return contract.validate_effect_counts(value, route="zero_effect_preflight")
    except contract.ContractViolation:
        _reject("preflight_nonzero_effect")


def _privacy_token_bytes(
    root: Path,
    provider: Callable[[Path], Sequence[bytes | str]],
) -> tuple[bytes, ...]:
    try:
        raw = provider(root)
    except Exception:
        _reject("preflight_privacy_tokens_unavailable")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        _reject("preflight_privacy_tokens_invalid")
    tokens: list[bytes] = []
    for item in raw:
        # bytes.lower() is reliably case-insensitive only for ASCII.  Refuse
        # an unscannable secret instead of silently weakening the privacy gate.
        if isinstance(item, str):
            try:
                token = item.encode("ascii")
            except UnicodeError:
                _reject("preflight_privacy_tokens_invalid")
        elif type(item) is bytes:
            token = item
        else:
            _reject("preflight_privacy_tokens_invalid")
        if not token or not token.isascii():
            _reject("preflight_privacy_tokens_invalid")
        tokens.append(token)
    if not tokens:
        _reject("preflight_privacy_tokens_invalid")
    return tuple(tokens)


def _scan_private_bytes(payload: bytes, forbidden: Sequence[bytes]) -> dict[str, bool]:
    if _payload_contains_forbidden_token(payload, forbidden):
        _reject("preflight_private_privacy_scan_failed")
    return {
        "forbidden_token_echo_absent": True,
        "absolute_private_path_absent": True,
        "readable_contact_absent": True,
    }


def _scan_public_bytes(payload: bytes, forbidden: Sequence[bytes]) -> dict[str, bool]:
    lowered = payload.lower()
    if (
        _payload_contains_forbidden_token(payload, forbidden)
        or _ACCESSION_RE.search(payload) is not None
        or b"http://" in lowered
        or b"https://" in lowered
        or b"canonical_request_body" in lowered
        or b"response_body" in lowered
        or b"normalized_text" in lowered
        or b"selected_filename" in lowered
    ):
        _reject("preflight_public_privacy_scan_failed")
    return {
        "forbidden_token_echo_absent": True,
        "accessions_absent": True,
        "urls_absent": True,
        "filenames_offsets_bodies_absent": True,
        "private_paths_absent": True,
        "pre_release_science_values_absent": True,
    }


def _json_strings(value: Any) -> list[str]:
    """Return every parsed JSON key and string value for privacy scanning."""

    strings: list[str] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is str:
            strings.append(item)
        elif type(item) is dict:
            for key, child in item.items():
                if type(key) is str:
                    strings.append(key)
                pending.append(child)
        elif type(item) is list:
            pending.extend(item)
    return strings


def _privacy_text_forms(value: str) -> frozenset[str]:
    """Normalize the spellings a private value can take inside JSON.

    In particular, Windows paths may be serialized with doubled backslashes,
    forward slashes, changed case, or ``\\uXXXX`` escapes.  Looking only for
    the original UTF-8 bytes would miss those ordinary JSON representations.
    """

    slash = value.replace("\\", "/")
    backslash = value.replace("/", "\\")
    raw = {value, slash, backslash}
    forms = {item.casefold() for item in raw if item}
    for item in raw:
        if not item:
            continue
        for ensure_ascii in (False, True):
            encoded = json.dumps(item, ensure_ascii=ensure_ascii)[1:-1]
            forms.add(encoded.casefold())
    return frozenset(forms)


def _payload_contains_forbidden_token(
    payload: bytes, forbidden: Sequence[bytes]
) -> bool:
    """Fail-closed token search over raw, JSON-escaped, and parsed strings."""

    if type(payload) is not bytes:
        return True
    byte_forms, text_forms = _forbidden_token_forms(forbidden)
    if not byte_forms:
        return True
    lowered = payload.lower()
    if any(item in lowered for item in byte_forms):
        return True
    try:
        parsed = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, TypeError):
        return False
    for candidate in _json_strings(parsed):
        candidate_forms = _privacy_text_forms(candidate)
        if any(
            token in candidate_form
            for candidate_form in candidate_forms
            for token in text_forms
        ):
            return True
    return False


def _forbidden_token_forms(
    forbidden: Sequence[bytes],
) -> tuple[frozenset[bytes], frozenset[str]]:
    byte_forms: set[bytes] = set()
    text_forms: set[str] = set()
    for token in forbidden:
        if type(token) is not bytes or not token:
            return frozenset(), frozenset()
        byte_forms.add(token.lower())
        try:
            decoded = token.decode("utf-8", errors="strict")
        except UnicodeError:
            continue
        for form in _privacy_text_forms(decoded):
            text_forms.add(form)
            byte_forms.add(form.encode("utf-8"))
    return frozenset(byte_forms), frozenset(text_forms)


def _stream_file_sha256_and_privacy(
    path: Path,
    *,
    byte_forms: frozenset[bytes],
    code: str,
) -> tuple[str, int]:
    """Hash a private file while scanning tokens across chunk boundaries."""

    maximum = max((len(item) for item in byte_forms), default=0)
    overlap = b""
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_TREE_SCAN_CHUNK_BYTES)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
                if byte_forms:
                    searchable = (overlap + chunk).lower()
                    if any(item in searchable for item in byte_forms):
                        _reject(code)
                    overlap = (
                        searchable[-(maximum - 1) :] if maximum > 1 else b""
                    )
    except V39PreflightError:
        raise
    except OSError:
        _reject(code)
    return digest.hexdigest(), byte_count


def _snapshot_serialized_tree(
    directory: Path,
    *,
    repo_root: Path,
    forbidden: Sequence[bytes] | None,
    code: str,
) -> dict[str, Any]:
    """Hash/stat a tree without following links or retaining file bodies."""

    try:
        resolved = directory.resolve(strict=True)
        resolved.relative_to(repo_root)
        root_details = directory.lstat()
    except (OSError, ValueError):
        _reject(code)
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or stat.S_ISLNK(root_details.st_mode)
        or getattr(root_details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    ):
        _reject(code)
    byte_forms: frozenset[bytes] = frozenset()
    if forbidden is not None:
        byte_forms, _text_forms = _forbidden_token_forms(forbidden)
        if not byte_forms:
            _reject(code)
    rows: list[dict[str, Any]] = []
    pending = [directory]
    while pending:
        parent = pending.pop()
        try:
            entries = sorted(os.scandir(parent), key=lambda item: item.name)
        except OSError:
            _reject(code)
        for entry in entries:
            path = Path(entry.path)
            try:
                details = entry.stat(follow_symlinks=False)
                relative = path.relative_to(directory).as_posix()
            except (OSError, ValueError):
                _reject(code)
            if (
                not relative
                or entry.is_symlink()
                or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            ):
                _reject(code)
            if forbidden is not None and _payload_contains_forbidden_token(
                relative.encode("utf-8", errors="strict"), forbidden
            ):
                _reject(code)
            common = {
                "path": relative,
                "mode": int(details.st_mode),
                "mtime_ns": int(details.st_mtime_ns),
            }
            if stat.S_ISDIR(details.st_mode):
                rows.append({**common, "kind": "directory"})
                pending.append(path)
            elif stat.S_ISREG(details.st_mode):
                digest, observed_bytes = _stream_file_sha256_and_privacy(
                    path, byte_forms=byte_forms, code=code
                )
                if observed_bytes != details.st_size:
                    _reject(code)
                rows.append(
                    {
                        **common,
                        "kind": "file",
                        "byte_count": observed_bytes,
                        "literal_sha256": digest,
                    }
                )
            else:
                _reject(code)
    rows.sort(key=lambda item: item["path"])
    return {
        "entry_count": len(rows),
        "inventory_sha256": contract.canonical_sha256(rows),
        "rows": rows,
    }


def _canonical_repo_root(value: Path) -> Path:
    if not isinstance(value, Path):
        _reject("preflight_repo_root_invalid")
    try:
        root = value.resolve(strict=True)
    except OSError:
        _reject("preflight_repo_root_invalid")
    if not root.is_dir() or not (root / ".git").exists():
        _reject("preflight_repo_root_invalid")
    return root


def _assert_regular_new_parent(path: Path, root: Path) -> None:
    try:
        resolved_parent = path.parent.resolve(strict=True)
        resolved_parent.relative_to(root)
        details = resolved_parent.lstat()
    except (OSError, ValueError):
        _reject("preflight_output_parent_invalid")
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        _reject("preflight_output_parent_invalid")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _optional_lstat(path: Path, *, code: str) -> os.stat_result | None:
    """Return an entry's own metadata without following a broken link."""

    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        _reject(code)


def _read_regular_file(
    path: Path, *, maximum: int | None, code: str
) -> bytes:
    """Read one ordinary, unlinked file and reject replacement races."""

    before = _optional_lstat(path, code=code)
    if before is None:
        _reject(code)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or getattr(before, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
        or before.st_nlink not in {0, 1}
        or before.st_size < 0
        or (maximum is not None and before.st_size > maximum)
    ):
        _reject(code)
    try:
        payload = path.read_bytes()
    except OSError:
        _reject(code)
    after = _optional_lstat(path, code=code)
    if (
        after is None
        or not stat.S_ISREG(after.st_mode)
        or stat.S_ISLNK(after.st_mode)
        or getattr(after, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
        or after.st_nlink not in {0, 1}
        or after.st_size != len(payload)
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        _reject(code)
    return payload


def _assert_exact_regular_file(path: Path, payload: bytes, *, code: str) -> None:
    if _read_regular_file(path, maximum=len(payload), code=code) != payload:
        _reject(code)


def _write_atomic_new_file(path: Path, payload: bytes, *, code: str) -> None:
    """Durably stage bytes and atomically promote them to a new public path."""

    pending = path.with_name(f"{path.name}.v39-pending")
    final_state = _optional_lstat(path, code=code)
    pending_state = _optional_lstat(pending, code=code)
    if final_state is not None:
        if pending_state is not None:
            _reject(code)
        _assert_exact_regular_file(path, payload, code=code)
        return

    if pending_state is None:
        try:
            with pending.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            # A simultaneous recovery may have staged the same immutable bytes.
            pass
        except OSError:
            _reject(code)
    _assert_exact_regular_file(pending, payload, code=code)
    _fsync_directory(path.parent)
    if _optional_lstat(path, code=code) is not None:
        _reject(code)
    try:
        os.replace(pending, path)
        _fsync_directory(path.parent)
    except OSError:
        _reject(code)
    _assert_exact_regular_file(path, payload, code=code)
    if _optional_lstat(pending, code=code) is not None:
        _reject(code)


def _write_new_file(path: Path, payload: bytes, *, code: str) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or details.st_nlink != 1
            or path.read_bytes() != payload
        ):
            _reject(code)
    except FileExistsError:
        _reject("preflight_already_consumed")
    except V39PreflightError:
        raise
    except OSError:
        _reject(code)


def _preflight_intent() -> dict[str, Any]:
    return _self_hash(
        {
            "schema_version": PREFLIGHT_INTENT_SCHEMA_VERSION,
            "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
            "contract_manifest_sha256": contract.CONTRACT_MANIFEST_SHA256,
            "preregistration_commit": contract.PREREGISTRATION_COMMIT,
            "external_effects_authorized": False,
        },
        "intent_sha256",
    )


def _reserve_once(root: Path) -> Path:
    public_path = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    if (
        _optional_lstat(public_path, code="preflight_already_consumed") is not None
        or _optional_lstat(private_root, code="preflight_already_consumed") is not None
    ):
        _reject("preflight_already_consumed")
    parent = private_root.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        parent_resolved = parent.resolve(strict=True)
        parent_resolved.relative_to(root)
        private_root.mkdir(exist_ok=False)
        (private_root / PRIVATE_MANIFEST_DIRECTORY).mkdir(exist_ok=False)
    except FileExistsError:
        _reject("preflight_already_consumed")
    except (OSError, ValueError):
        _reject("preflight_private_state_unwritable")
    intent = _preflight_intent()
    _write_new_file(
        private_root / PRIVATE_INTENT_FILENAME,
        _artifact_bytes(intent),
        code="preflight_private_state_unwritable",
    )
    _fsync_directory(parent_resolved)
    return private_root


def _projection_manifest(value: Any) -> dict[str, Any]:
    try:
        candidate = getattr(value, "manifest")
    except Exception:
        _reject("preflight_projection_invalid")
    return validate_bridge_manifest(candidate)


def _build_private_manifest(
    *,
    repository: Mapping[str, Any],
    bridge: Mapping[str, Any],
    commitments: Mapping[str, Any],
    effect_before: Mapping[str, int],
    effect_after: Mapping[str, int],
    tests: Mapping[str, Any],
) -> dict[str, Any]:
    gates = {
        "implementation_ancestry_passed": True,
        "exact_twelve_file_delta_passed": True,
        "predecessor_blobs_unchanged": True,
        "source_authority_authenticated": True,
        "streaming_projection_parity_passed": True,
        "request_and_pilot_commitments_frozen": True,
        "confirmation_and_final_unreachable": True,
        "zero_external_effects": True,
        "offline_tests_passed": True,
        "privacy_passed": True,
        "eligible_for_public_seal": True,
    }
    unsigned = {
        "schema_version": PRIVATE_MANIFEST_SCHEMA_VERSION,
        "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
        "status": "passed",
        "contract_manifest_sha256": contract.CONTRACT_MANIFEST_SHA256,
        "repository": copy.deepcopy(dict(repository)),
        "bridge": copy.deepcopy(dict(bridge)),
        "request_commitments": copy.deepcopy(dict(commitments)),
        "effect_counts_before": copy.deepcopy(dict(effect_before)),
        "effect_counts_after": copy.deepcopy(dict(effect_after)),
        "offline_tests": copy.deepcopy(dict(tests)),
        "privacy": {
            "aggregate_only_bridge_serialized": True,
            "aggregate_only_request_commitments_serialized": True,
            "readable_contact_stored": False,
            "absolute_private_path_stored": False,
        },
        "gates": gates,
    }
    return _self_hash(unsigned, "private_manifest_sha256")


def _build_public_artifact(
    *,
    repository: Mapping[str, Any],
    bridge: Mapping[str, Any],
    commitments: Mapping[str, Any],
    effects: Mapping[str, int],
    tests: Mapping[str, Any],
    private_manifest_sha256: str,
    private_manifest_literal_sha256: str,
) -> dict[str, Any]:
    ancestry = repository["ancestry"]
    gates = {
        "implementation_ancestry_passed": True,
        "exact_twelve_file_delta_passed": True,
        "predecessor_blobs_unchanged": True,
        "source_authority_authenticated": True,
        "streaming_projection_parity_passed": True,
        "request_and_pilot_commitments_frozen": True,
        "confirmation_and_final_unreachable": True,
        "zero_external_effects": True,
        "offline_tests_passed": True,
        "privacy_passed": True,
    }
    unsigned = {
        "schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
        "status": "passed",
        "stage": contract.DEVELOPMENT_COMMAND,
        "branch": contract.BRANCH_NAME,
        "preregistration_commit": contract.PREREGISTRATION_COMMIT,
        "preregistration_tree": contract.PREREGISTRATION_TREE,
        "implementation_commit": ancestry["commit"],
        "implementation_tree": ancestry["tree"],
        "contract_manifest_sha256": contract.CONTRACT_MANIFEST_SHA256,
        "science_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
        "source_authority": {
            "terminal_internal_sha256": contract.V38_TERMINAL_INTERNAL_SHA256,
            "source_authority_literal_sha256": (
                contract.V38_SOURCE_AUTHORITY_LITERAL_SHA256
            ),
            "logical_checkpoint_sha256": contract.V38_LOGICAL_CHECKPOINT_SHA256,
            "stage_source_seal_sha256": contract.V38_STAGE_SOURCE_SEAL_SHA256,
            "compact_replay_sha256": contract.V38_COMPACT_REPLAY_SHA256,
            "role_manifest_inventory_sha256": (
                contract.V38_ROLE_MANIFEST_INVENTORY_SHA256
            ),
            "role_plan_sha256": contract.V38_ROLE_PLAN_SHA256,
            "inventory_sha256": contract.V38_INVENTORY_SHA256,
        },
        "counts": {
            "documents": commitments["document_count"],
            "records": commitments["record_count"],
            "events": commitments["event_count"],
            "canonical_requests": commitments["request_count"],
            "pilots": commitments["pilot_count"],
            "remaining": commitments["remaining_count"],
            "implementation_production_files": len(
                contract.IMPLEMENTATION_PRODUCTION_PATHS
            ),
            "implementation_test_files": len(contract.IMPLEMENTATION_TEST_PATHS),
            "experiment_family_sec_requests": (
                contract.EXPERIMENT_FAMILY_SEC_REQUEST_COUNT
            ),
        },
        "aggregates": {
            "bridge_sha256": bridge["bridge_sha256"],
            "legacy_projection_sha256": bridge["legacy_projection_sha256"],
            "documents_sha256": commitments["documents_sha256"],
            "records_sha256": commitments["records_sha256"],
            "events_sha256": commitments["events_sha256"],
            "preprocessed_events_sha256": commitments[
                "preprocessed_events_sha256"
            ],
            "canonical_requests_sha256": commitments[
                "canonical_requests_sha256"
            ],
            "pilot_order_sha256": commitments["pilot_order_sha256"],
            "remaining_order_sha256": commitments["remaining_order_sha256"],
            "source_order_sha256": commitments["source_order_sha256"],
            "prior_links_sha256": commitments["prior_links_sha256"],
            "production_source_inventory_sha256": repository[
                "production_source_inventory_sha256"
            ],
            "test_source_inventory_sha256": repository[
                "test_source_inventory_sha256"
            ],
            "all_source_inventory_sha256": repository[
                "source_inventory_sha256"
            ],
            "predecessor_inventory_sha256": repository[
                "predecessor_inventory_sha256"
            ],
        },
        "effect_counts": copy.deepcopy(dict(effects)),
        "offline_tests": copy.deepcopy(dict(tests)),
        "privacy": {
            "readable_contact_published": False,
            "accessions_urls_filenames_offsets_bodies_published": False,
            "private_paths_published": False,
            "canonical_requests_published": False,
            "pre_release_science_values_published": False,
            "redacted_errors_only": True,
        },
        "gates": gates,
        "private_manifest_sha256": private_manifest_sha256,
        "private_manifest_literal_sha256": private_manifest_literal_sha256,
        "eligible_for_public_seal": True,
        "development_authorized": False,
    }
    return _self_hash(unsigned, "public_artifact_sha256")


def validate_private_preflight_manifest(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _PRIVATE_MANIFEST_FIELDS,
        code="preflight_private_manifest_shape_invalid",
    )
    _validate_self_hash(item, "private_manifest_sha256", code="preflight_private_manifest_hash_invalid")
    gates = _strict_mapping(
        item["gates"], _PRIVATE_GATES, code="preflight_private_manifest_invalid"
    )
    privacy = _strict_mapping(
        item["privacy"],
        _PRIVATE_PRIVACY_FIELDS,
        code="preflight_private_manifest_invalid",
    )
    if (
        item["schema_version"] != PRIVATE_MANIFEST_SCHEMA_VERSION
        or item["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or item["status"] != "passed"
        or item["contract_manifest_sha256"] != contract.CONTRACT_MANIFEST_SHA256
        or any(child is not True for child in gates.values())
        or privacy
        != {
            "aggregate_only_bridge_serialized": True,
            "aggregate_only_request_commitments_serialized": True,
            "readable_contact_stored": False,
            "absolute_private_path_stored": False,
        }
    ):
        _reject("preflight_private_manifest_invalid")
    validate_repository_snapshot(item["repository"])
    bridge = validate_bridge_manifest(item["bridge"])
    commitments = validate_request_commitments(item["request_commitments"])
    _validate_bridge_commitment_parity(bridge, commitments)
    _validate_effect_snapshot(item["effect_counts_before"])
    _validate_effect_snapshot(item["effect_counts_after"])
    validate_offline_test_report(item["offline_tests"])
    return item


def validate_public_preflight_artifact(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _PUBLIC_ARTIFACT_FIELDS,
        code="preflight_public_artifact_shape_invalid",
    )
    _validate_self_hash(item, "public_artifact_sha256", code="preflight_public_artifact_hash_invalid")
    source_authority = _strict_mapping(
        item["source_authority"],
        frozenset(
            {
                "terminal_internal_sha256",
                "source_authority_literal_sha256",
                "logical_checkpoint_sha256",
                "stage_source_seal_sha256",
                "compact_replay_sha256",
                "role_manifest_inventory_sha256",
                "role_plan_sha256",
                "inventory_sha256",
            }
        ),
        code="preflight_public_artifact_invalid",
    )
    counts = _strict_mapping(
        item["counts"],
        frozenset(
            {
                "documents",
                "records",
                "events",
                "canonical_requests",
                "pilots",
                "remaining",
                "implementation_production_files",
                "implementation_test_files",
                "experiment_family_sec_requests",
            }
        ),
        code="preflight_public_artifact_invalid",
    )
    aggregates = _strict_mapping(
        item["aggregates"],
        frozenset(
            {
                "bridge_sha256",
                "legacy_projection_sha256",
                "documents_sha256",
                "records_sha256",
                "events_sha256",
                "preprocessed_events_sha256",
                "canonical_requests_sha256",
                "pilot_order_sha256",
                "remaining_order_sha256",
                "source_order_sha256",
                "prior_links_sha256",
                "production_source_inventory_sha256",
                "test_source_inventory_sha256",
                "all_source_inventory_sha256",
                "predecessor_inventory_sha256",
            }
        ),
        code="preflight_public_artifact_invalid",
    )
    privacy = _strict_mapping(
        item["privacy"],
        _PUBLIC_PRIVACY_FIELDS,
        code="preflight_public_artifact_invalid",
    )
    gates = _strict_mapping(
        item["gates"], _PUBLIC_GATES, code="preflight_public_artifact_invalid"
    )
    expected_source = {
        "terminal_internal_sha256": contract.V38_TERMINAL_INTERNAL_SHA256,
        "source_authority_literal_sha256": contract.V38_SOURCE_AUTHORITY_LITERAL_SHA256,
        "logical_checkpoint_sha256": contract.V38_LOGICAL_CHECKPOINT_SHA256,
        "stage_source_seal_sha256": contract.V38_STAGE_SOURCE_SEAL_SHA256,
        "compact_replay_sha256": contract.V38_COMPACT_REPLAY_SHA256,
        "role_manifest_inventory_sha256": contract.V38_ROLE_MANIFEST_INVENTORY_SHA256,
        "role_plan_sha256": contract.V38_ROLE_PLAN_SHA256,
        "inventory_sha256": contract.V38_INVENTORY_SHA256,
    }
    expected_counts = {
        "documents": contract.DEVELOPMENT_DOCUMENT_COUNT,
        "records": contract.DEVELOPMENT_DOCUMENT_COUNT,
        "events": contract.DEVELOPMENT_DOCUMENT_COUNT,
        "canonical_requests": contract.DEVELOPMENT_DOCUMENT_COUNT,
        "pilots": contract.DEVELOPMENT_PILOT_COUNT,
        "remaining": contract.DEVELOPMENT_REMAINING_COUNT,
        "implementation_production_files": len(contract.IMPLEMENTATION_PRODUCTION_PATHS),
        "implementation_test_files": len(contract.IMPLEMENTATION_TEST_PATHS),
        "experiment_family_sec_requests": contract.EXPERIMENT_FAMILY_SEC_REQUEST_COUNT,
    }
    expected_privacy = {
        "readable_contact_published": False,
        "accessions_urls_filenames_offsets_bodies_published": False,
        "private_paths_published": False,
        "canonical_requests_published": False,
        "pre_release_science_values_published": False,
        "redacted_errors_only": True,
    }
    if (
        item["schema_version"] != PUBLIC_ARTIFACT_SCHEMA_VERSION
        or item["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or item["status"] != "passed"
        or item["stage"] != contract.DEVELOPMENT_COMMAND
        or item["branch"] != contract.BRANCH_NAME
        or item["preregistration_commit"] != contract.PREREGISTRATION_COMMIT
        or item["preregistration_tree"] != contract.PREREGISTRATION_TREE
        or not _is_sha1(item["implementation_commit"])
        or not _is_sha1(item["implementation_tree"])
        or item["contract_manifest_sha256"] != contract.CONTRACT_MANIFEST_SHA256
        or item["science_projection_sha256"] != contract.SCIENCE_PROJECTION_SHA256
        or not _is_sha256(item["private_manifest_sha256"])
        or not _is_sha256(item["private_manifest_literal_sha256"])
        or item["eligible_for_public_seal"] is not True
        or item["development_authorized"] is not False
        or source_authority != expected_source
        or counts != expected_counts
        or any(not _is_sha256(child) for child in aggregates.values())
        or privacy != expected_privacy
        or any(child is not True for child in gates.values())
    ):
        _reject("preflight_public_artifact_invalid")
    _validate_effect_snapshot(item["effect_counts"])
    validate_offline_test_report(item["offline_tests"])
    return item


def validate_public_failed_preflight_artifact(value: Any) -> dict[str, Any]:
    """Validate the fixed redacted terminal receipt for a consumed failure."""

    item = _strict_mapping(
        value,
        _PUBLIC_FAILED_PREFLIGHT_FIELDS,
        code="preflight_public_failure_shape_invalid",
    )
    _validate_self_hash(
        item,
        "public_artifact_sha256",
        code="preflight_public_failure_hash_invalid",
    )
    if (
        item["schema_version"] != PUBLIC_FAILED_PREFLIGHT_SCHEMA_VERSION
        or item["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or item["status"] != "failed"
        or item["stage"] != contract.DEVELOPMENT_COMMAND
        or item["branch"] != contract.BRANCH_NAME
        or type(item["failure_code"]) is not str
        or re.fullmatch(r"[a-z0-9_]{3,80}", item["failure_code"]) is None
        or item["preflight_consumed"] is not True
        or item["failure_preserved"] is not True
        or item["redacted_error_only"] is not True
        or item["eligible_for_public_seal"] is not False
        or item["rerun_authorized"] is not False
        or item["development_authorized"] is not False
        or item["confirmation_and_final_opened"] is not False
        or item["real_money_authorized"] is not False
    ):
        _reject("preflight_public_failure_invalid")
    return item


def _recover_failed_preflight_publication(root: Path) -> None:
    """Finish only an exact failed-preflight pending promotion after a crash."""

    target = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
    pending = target.with_name(f"{target.name}.v39-pending")
    final_state = _optional_lstat(
        target, code="preflight_failure_preservation_failed"
    )
    pending_state = _optional_lstat(
        pending, code="preflight_failure_preservation_failed"
    )
    if pending_state is None:
        return
    if final_state is not None:
        _reject("preflight_failure_preservation_failed")

    _assert_regular_new_parent(target, root)
    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    private_state = _optional_lstat(
        private_root, code="preflight_failure_preservation_failed"
    )
    if (
        private_state is None
        or not stat.S_ISDIR(private_state.st_mode)
        or stat.S_ISLNK(private_state.st_mode)
        or getattr(private_state, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
    ):
        _reject("preflight_failure_preservation_failed")
    try:
        private_root.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        _reject("preflight_failure_preservation_failed")
    _assert_exact_regular_file(
        private_root / PRIVATE_INTENT_FILENAME,
        _artifact_bytes(_preflight_intent()),
        code="preflight_failure_preservation_failed",
    )

    payload = _read_regular_file(
        pending,
        maximum=64 * 1024,
        code="preflight_failure_preservation_failed",
    )
    artifact = validate_public_failed_preflight_artifact(
        _parse_artifact_bytes(
            payload,
            maximum=64 * 1024,
            code="preflight_failure_preservation_failed",
        )
    )
    if _artifact_bytes(artifact) != payload:
        _reject("preflight_failure_preservation_failed")
    _write_atomic_new_file(
        target, payload, code="preflight_failure_preservation_failed"
    )


def _preserve_failed_preflight(
    root: Path, *, private_root: Path, failure_code: str
) -> dict[str, Any]:
    """Publish one redacted failure after the private intent was consumed."""

    try:
        private_root.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        _reject("preflight_failure_preservation_failed")
    body = {
        "schema_version": PUBLIC_FAILED_PREFLIGHT_SCHEMA_VERSION,
        "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
        "status": "failed",
        "stage": contract.DEVELOPMENT_COMMAND,
        "branch": contract.BRANCH_NAME,
        "failure_code": failure_code,
        "preflight_consumed": True,
        "failure_preserved": True,
        "redacted_error_only": True,
        "eligible_for_public_seal": False,
        "rerun_authorized": False,
        "development_authorized": False,
        "confirmation_and_final_opened": False,
        "real_money_authorized": False,
    }
    artifact = validate_public_failed_preflight_artifact(
        _self_hash(body, "public_artifact_sha256")
    )
    payload = _artifact_bytes(artifact)
    safe_tokens = _privacy_token_bytes(
        root,
        lambda path: (
            str(path),
            str(path).replace("\\", "/"),
            str((path / V38_PRIVATE_ROOT).resolve(strict=False)),
            str(
                (path / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False)
            ),
        ),
    )
    _scan_public_bytes(payload, safe_tokens)
    target = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        _reject("preflight_failure_preservation_failed")
    _assert_regular_new_parent(target, root)
    _write_atomic_new_file(
        target, payload, code="preflight_failure_preservation_failed"
    )
    return artifact


def run_preflight(
    repo_root: Path,
    *,
    dependencies: PreflightDependencies | None = None,
) -> dict[str, Any]:
    """Consume and seal the v3.9 preflight exactly once, without external effects."""

    root = _canonical_repo_root(repo_root)
    _recover_failed_preflight_publication(root)
    deps = dependencies if dependencies is not None else build_production_dependencies()
    if type(deps) is not PreflightDependencies:
        _reject("preflight_dependencies_invalid")
    private_root = _reserve_once(root)
    try:
        before_effects = _validate_effect_snapshot(deps.effect_snapshot())
        repository = validate_repository_snapshot(deps.inspect_repository(root))
        authority = deps.authenticate_source(root)
        projection = deps.build_projection(authority)
        bridge = _projection_manifest(projection)
        commitments = validate_request_commitments(
            deps.build_request_commitments(projection)
        )
        _validate_bridge_commitment_parity(bridge, commitments)
        tests = validate_offline_test_report(deps.run_offline_tests(root))
        after_effects = _validate_effect_snapshot(deps.effect_snapshot())
        if before_effects != after_effects:
            _reject("preflight_effect_accounting_changed")
        tokens = _privacy_token_bytes(root, deps.privacy_tokens)

        private_manifest = _build_private_manifest(
            repository=repository,
            bridge=bridge,
            commitments=commitments,
            effect_before=before_effects,
            effect_after=after_effects,
            tests=tests,
        )
        validated_private = validate_private_preflight_manifest(private_manifest)
        private_payload = _artifact_bytes(validated_private)
        private_privacy = _scan_private_bytes(private_payload, tokens)
        if not all(private_privacy.values()):
            _reject("preflight_private_privacy_scan_failed")
        private_literal_sha256 = hashlib.sha256(private_payload).hexdigest()
        private_target = (
            private_root
            / PRIVATE_MANIFEST_DIRECTORY
            / f"{private_literal_sha256}.json"
        )
        _write_new_file(
            private_target,
            private_payload,
            code="preflight_private_manifest_unwritable",
        )

        public_artifact = _build_public_artifact(
            repository=repository,
            bridge=bridge,
            commitments=commitments,
            effects=after_effects,
            tests=tests,
            private_manifest_sha256=validated_private["private_manifest_sha256"],
            private_manifest_literal_sha256=private_literal_sha256,
        )
        validated_public = validate_public_preflight_artifact(public_artifact)
        public_payload = _artifact_bytes(validated_public)
        public_privacy = _scan_public_bytes(public_payload, tokens)
        if not all(public_privacy.values()):
            _reject("preflight_public_privacy_scan_failed")
        # Seal private success before the public file.  The public write is the
        # final fallible success step, so a preceding failure can still publish
        # the one allowed redacted failed artifact at that path.
        completion = _self_hash(
            {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "preflight_attempt_id": PREFLIGHT_ATTEMPT_ID,
                "private_manifest_sha256": validated_private[
                    "private_manifest_sha256"
                ],
                "private_manifest_literal_sha256": private_literal_sha256,
                "public_artifact_sha256": validated_public[
                    "public_artifact_sha256"
                ],
                "public_artifact_literal_sha256": hashlib.sha256(
                    public_payload
                ).hexdigest(),
                "status": "passed",
            },
            "completion_sha256",
        )
        _write_new_file(
            private_root / PRIVATE_COMPLETION_FILENAME,
            _artifact_bytes(completion),
            code="preflight_completion_unwritable",
        )
        public_target = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
        try:
            public_target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            _reject("preflight_public_artifact_unwritable")
        _assert_regular_new_parent(public_target, root)
        _write_new_file(
            public_target,
            public_payload,
            code="preflight_public_artifact_unwritable",
        )
        return validated_public
    except V39PreflightError as exc:
        failure_code = exc.code
    except Exception:
        failure_code = "preflight_dependency_failed"
    _preserve_failed_preflight(
        root, private_root=private_root, failure_code=failure_code
    )
    raise V39PreflightError(failure_code)


def _git(root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        _reject("preflight_git_failed")
    if completed.returncode != 0:
        _reject("preflight_git_failed")
    if binary:
        return completed.stdout
    try:
        return completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError:
        _reject("preflight_git_failed")


def _git_tree_rows(root: Path, revision: str) -> list[dict[str, str]]:
    text = _git(root, "ls-tree", "-r", "--full-tree", revision)
    if type(text) is not str:
        _reject("preflight_git_failed")
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        try:
            left, path = line.split("\t", 1)
            mode, kind, object_id = left.split(" ", 2)
        except ValueError:
            _reject("preflight_git_failed")
        if (
            not path
            or "\n" in path
            or "\r" in path
            or not _is_sha1(object_id)
        ):
            _reject("preflight_git_failed")
        rows.append(
            {"path": path, "mode": mode, "kind": kind, "object_id": object_id}
        )
    return rows


def inspect_pushed_implementation(repo_root: Path) -> dict[str, Any]:
    """Inspect local public Git state without fetching or changing it."""

    root = _canonical_repo_root(repo_root)
    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    parent = _git(root, "rev-parse", "HEAD^")
    remote = _git(root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    changed_text = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        parent,
        commit,
    )
    if not all(type(item) is str for item in (branch, commit, tree, parent, remote, status, changed_text)):
        _reject("preflight_git_failed")
    changed: dict[str, str] = {}
    for line in changed_text.splitlines():
        try:
            state, path = line.split("\t", 1)
        except ValueError:
            _reject("preflight_git_delta_invalid")
        if path in changed:
            _reject("preflight_git_delta_invalid")
        changed[path] = state
    ancestry = {
        "branch": branch,
        "commit": commit,
        "tree": tree,
        "parent": parent,
        "local_head": commit,
        "remote_head": remote,
        "changed_paths": changed,
        "clean_worktree": status == "",
        "preregistration_authenticated": True,
        "predecessor_blobs_unchanged": True,
    }
    try:
        contract.validate_implementation_ancestry(ancestry)
    except contract.ContractViolation:
        _reject("preflight_implementation_ancestry_invalid")

    preregistration = _git(
        root,
        "show",
        f"{contract.PREREGISTRATION_COMMIT}:{contract.PREREGISTRATION_PATH}",
        binary=True,
    )
    prereg_blob = _git(
        root,
        "rev-parse",
        f"{contract.PREREGISTRATION_COMMIT}:{contract.PREREGISTRATION_PATH}",
    )
    if (
        type(preregistration) is not bytes
        or type(prereg_blob) is not str
        or prereg_blob != contract.PREREGISTRATION_GIT_BLOB_SHA1
        or len(preregistration) != contract.PREREGISTRATION_LITERAL_BYTES
        or hashlib.sha256(preregistration).hexdigest()
        != contract.PREREGISTRATION_LITERAL_SHA256
    ):
        _reject("preflight_preregistration_invalid")

    source_rows: list[dict[str, Any]] = []
    for path in contract.IMPLEMENTATION_ALLOWED_PATHS:
        blob = _git(root, "show", f"{commit}:{path}", binary=True)
        blob_sha1 = _git(root, "rev-parse", f"{commit}:{path}")
        if type(blob) is not bytes or type(blob_sha1) is not str:
            _reject("preflight_source_inventory_invalid")
        try:
            working = (root / Path(path)).read_bytes()
        except OSError:
            _reject("preflight_source_inventory_invalid")
        if working != blob:
            _reject("preflight_source_inventory_invalid")
        source_rows.append(
            {
                "path": path,
                "git_blob_sha1": blob_sha1,
                "literal_sha256": hashlib.sha256(blob).hexdigest(),
                "byte_count": len(blob),
            }
        )
    parent_rows = _git_tree_rows(root, contract.PREREGISTRATION_COMMIT)
    head_rows = _git_tree_rows(root, commit)
    predecessor_paths = {row["path"] for row in parent_rows}
    unchanged_rows = [row for row in head_rows if row["path"] in predecessor_paths]
    predecessor_hash = contract.canonical_sha256(parent_rows)
    unchanged_hash = contract.canonical_sha256(unchanged_rows)
    snapshot = {
        "ancestry": ancestry,
        "source_inventory": source_rows,
        "source_inventory_sha256": contract.canonical_sha256(source_rows),
        "production_source_inventory_sha256": contract.canonical_sha256(
            source_rows[: len(contract.IMPLEMENTATION_PRODUCTION_PATHS)]
        ),
        "test_source_inventory_sha256": contract.canonical_sha256(
            source_rows[len(contract.IMPLEMENTATION_PRODUCTION_PATHS) :]
        ),
        "predecessor_inventory_sha256": predecessor_hash,
        "unchanged_predecessor_inventory_sha256": unchanged_hash,
        "preregistration_authenticated": True,
    }
    return validate_repository_snapshot(snapshot)


def _parse_artifact_bytes(payload: bytes, *, maximum: int, code: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or len(payload) > maximum:
        _reject(code)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if type(key) is not str or key in result:
                _reject(code)
            result[key] = child
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _reject(code),
        )
    except V39PreflightError:
        raise
    except (TypeError, ValueError, UnicodeError):
        _reject(code)
    if not isinstance(value, Mapping) or _artifact_bytes(value) != payload:
        _reject(code)
    return copy.deepcopy(dict(value))


def load_private_contact(repo_root: Path) -> str:
    """Load the ignored SEC contact without logging, hashing, or persisting it."""

    return _load_private_contact(_canonical_repo_root(repo_root))


def _read_execution_private_state(
    root: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    private_root = root / Path(contract.PRIVATE_PREFLIGHT_NAMESPACE)
    try:
        resolved = private_root.resolve(strict=True)
        resolved.relative_to(root)
        details = resolved.lstat()
        names = {item.name for item in resolved.iterdir()}
    except (OSError, ValueError):
        _reject("execution_preflight_private_state_invalid")
    if (
        not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or names
        != {
            PRIVATE_INTENT_FILENAME,
            PRIVATE_MANIFEST_DIRECTORY,
            PRIVATE_COMPLETION_FILENAME,
        }
    ):
        _reject("execution_preflight_private_state_invalid")

    try:
        intent_payload = (resolved / PRIVATE_INTENT_FILENAME).read_bytes()
        completion_payload = (resolved / PRIVATE_COMPLETION_FILENAME).read_bytes()
        manifests = list((resolved / PRIVATE_MANIFEST_DIRECTORY).iterdir())
    except OSError:
        _reject("execution_preflight_private_state_invalid")
    intent = _parse_artifact_bytes(
        intent_payload,
        maximum=64 * 1024,
        code="execution_preflight_intent_invalid",
    )
    if set(intent) != {
        "schema_version",
        "preflight_attempt_id",
        "contract_manifest_sha256",
        "preregistration_commit",
        "external_effects_authorized",
        "intent_sha256",
    }:
        _reject("execution_preflight_intent_invalid")
    _validate_self_hash(
        intent, "intent_sha256", code="execution_preflight_intent_invalid"
    )
    if (
        intent["schema_version"] != PREFLIGHT_INTENT_SCHEMA_VERSION
        or intent["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or intent["contract_manifest_sha256"] != contract.CONTRACT_MANIFEST_SHA256
        or intent["preregistration_commit"] != contract.PREREGISTRATION_COMMIT
        or intent["external_effects_authorized"] is not False
    ):
        _reject("execution_preflight_intent_invalid")

    if len(manifests) != 1 or not manifests[0].is_file():
        _reject("execution_preflight_private_manifest_invalid")
    manifest_path = manifests[0]
    try:
        manifest_payload = manifest_path.read_bytes()
    except OSError:
        _reject("execution_preflight_private_manifest_invalid")
    manifest_literal = hashlib.sha256(manifest_payload).hexdigest()
    if manifest_path.name != f"{manifest_literal}.json":
        _reject("execution_preflight_private_manifest_invalid")
    manifest = validate_private_preflight_manifest(
        _parse_artifact_bytes(
            manifest_payload,
            maximum=4 * 1024 * 1024,
            code="execution_preflight_private_manifest_invalid",
        )
    )

    completion = _parse_artifact_bytes(
        completion_payload,
        maximum=64 * 1024,
        code="execution_preflight_completion_invalid",
    )
    if set(completion) != {
        "schema_version",
        "preflight_attempt_id",
        "private_manifest_sha256",
        "private_manifest_literal_sha256",
        "public_artifact_sha256",
        "public_artifact_literal_sha256",
        "status",
        "completion_sha256",
    }:
        _reject("execution_preflight_completion_invalid")
    _validate_self_hash(
        completion,
        "completion_sha256",
        code="execution_preflight_completion_invalid",
    )
    if (
        completion["schema_version"] != PREFLIGHT_SCHEMA_VERSION
        or completion["preflight_attempt_id"] != PREFLIGHT_ATTEMPT_ID
        or completion["private_manifest_sha256"]
        != manifest["private_manifest_sha256"]
        or completion["private_manifest_literal_sha256"] != manifest_literal
        or not _is_sha256(completion["public_artifact_sha256"])
        or not _is_sha256(completion["public_artifact_literal_sha256"])
        or completion["status"] != "passed"
    ):
        _reject("execution_preflight_completion_invalid")
    return manifest, manifest_payload, completion, completion_payload


def _changed_paths(root: Path, parent: str, child: str) -> dict[str, str]:
    changed_text = _git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        parent,
        child,
    )
    if type(changed_text) is not str:
        _reject("execution_preflight_git_invalid")
    changed: dict[str, str] = {}
    for line in changed_text.splitlines():
        try:
            state, path = line.split("\t", 1)
        except ValueError:
            _reject("execution_preflight_git_invalid")
        if path in changed:
            _reject("execution_preflight_git_invalid")
        changed[path] = state
    return changed


def _parse_canonical_mapping_bytes(
    payload: bytes, *, maximum: int, code: str
) -> dict[str, Any]:
    """Parse duplicate-free canonical JSON that has no trailing newline."""

    if type(payload) is not bytes or not payload or len(payload) > maximum:
        _reject(code)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if type(key) is not str or key in result:
                _reject(code)
            result[key] = child
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: _reject(code),
        )
    except V39PreflightError:
        raise
    except (TypeError, ValueError, UnicodeError):
        _reject(code)
    if not isinstance(value, Mapping):
        _reject(code)
    try:
        encoded = contract.canonical_json_bytes(value)
    except Exception:
        _reject(code)
    if encoded != payload:
        _reject(code)
    return copy.deepcopy(dict(value))


def _git_single_parent(root: Path, revision: str, *, code: str) -> str:
    value = _git(root, "rev-list", "--parents", "-n", "1", revision)
    if type(value) is not str:
        _reject(code)
    fields = value.split()
    if len(fields) != 2 or fields[0] != revision or not _is_sha1(fields[1]):
        _reject(code)
    return fields[1]


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _git_blob_material(
    root: Path, revision: str, path: str, *, maximum: int, code: str
) -> dict[str, Any]:
    payload = _git(root, "show", f"{revision}:{path}", binary=True)
    object_id = _git(root, "rev-parse", f"{revision}:{path}")
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > maximum
        or type(object_id) is not str
        or not _is_sha1(object_id)
        or _git_blob_sha1(payload) != object_id
    ):
        _reject(code)
    return {
        "path": path,
        "git_blob_sha1": object_id,
        "literal_sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "payload": payload,
    }


def _inspect_result_git(root: Path) -> dict[str, Any]:
    """Authenticate the pushed R shell before any private value is loaded."""

    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    remote = _git(
        root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}"
    )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if not all(type(item) is str for item in (branch, commit, remote, status, tree)):
        _reject("pushed_result_git_invalid")
    if (
        branch != contract.BRANCH_NAME
        or not _is_sha1(commit)
        or not _is_sha1(tree)
        or commit != remote
        or status != ""
    ):
        _reject("pushed_result_git_invalid")
    parent = _git_single_parent(root, commit, code="pushed_result_ancestry_invalid")
    changed = _changed_paths(root, parent, commit)
    if changed != {
        contract.RESULT_ARTIFACT_PATH: "A",
        contract.COMPARISON_PATH: "M",
    }:
        _reject("pushed_result_delta_invalid")
    result_blob = _git_blob_material(
        root,
        commit,
        contract.RESULT_ARTIFACT_PATH,
        maximum=MAX_RESULT_ARTIFACT_BYTES,
        code="pushed_result_artifact_blob_invalid",
    )
    comparison_blob = _git_blob_material(
        root,
        commit,
        contract.COMPARISON_PATH,
        maximum=MAX_COMPARISON_BYTES,
        code="pushed_result_comparison_blob_invalid",
    )
    try:
        working_result = (root / Path(contract.RESULT_ARTIFACT_PATH)).read_bytes()
        working_comparison = (root / Path(contract.COMPARISON_PATH)).read_bytes()
    except OSError:
        _reject("pushed_result_worktree_invalid")
    if (
        working_result != result_blob["payload"]
        or working_comparison != comparison_blob["payload"]
    ):
        _reject("pushed_result_worktree_invalid")
    return {
        "branch": branch,
        "commit": commit,
        "tree": tree,
        "parent": parent,
        "remote": remote,
        "status": status,
        "changed_paths": changed,
        "result_blob": result_blob,
        "comparison_blob": comparison_blob,
    }


def _validate_replayed_pilot_guard(value: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _PUSHED_RESULT_PILOT_GUARD_FIELDS,
        code="pushed_result_pause_artifact_invalid",
    )
    try:
        item = contract.validate_self_sha256(item, field="segment_guard_sha256")
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    events = item["ordered_generation_response_event_sha256s"]
    if (
        item["schema_version"] != PUSHED_RESULT_RUNTIME_GUARD_SCHEMA_VERSION
        or item["segment_id"] != "pilot"
        or item["store_segment_id"] != "initial"
        or item["generation_count"] != contract.DEVELOPMENT_PILOT_COUNT
        or not all(
            _is_sha256(item[field])
            for field in (
                "pre_runtime_receipt_sha256",
                "post_runtime_receipt_sha256",
                "stable_runtime_identity_sha256",
            )
        )
        or type(events) is not list
        or len(events) != contract.DEVELOPMENT_PILOT_COUNT
        or any(not _is_sha256(child) for child in events)
        or item["ordered_generation_response_events_sha256"]
        != contract.canonical_sha256(events)
        or item["raw_show_hash_is_diagnostic_only"] is not True
        or item["modified_at_is_excluded_only"] is not True
        or item["identity_http_request_count"] != 4
        or item["retry_count"] != 0
    ):
        _reject("pushed_result_pause_artifact_invalid")
    return item


def _validate_replayed_pilot_latency(value: Any, *, plan: Any) -> dict[str, Any]:
    item = _strict_mapping(
        value,
        _PUSHED_RESULT_LATENCY_FIELDS,
        code="pushed_result_pause_artifact_invalid",
    )
    try:
        item = contract.validate_self_sha256(
            item, field="latency_receipt_sha256"
        )
        execution_calls = list(getattr(plan, "execution_calls"))
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    rows_raw = item["pilot_rows"]
    if (
        type(rows_raw) is not list
        or len(rows_raw) != contract.DEVELOPMENT_PILOT_COUNT
        or len(execution_calls) != contract.DEVELOPMENT_DOCUMENT_COUNT
    ):
        _reject("pushed_result_pause_artifact_invalid")
    rows: list[dict[str, Any]] = []
    durations: list[int] = []
    for index, raw in enumerate(rows_raw):
        row = _strict_mapping(
            raw,
            _PUSHED_RESULT_LATENCY_ROW_FIELDS,
            code="pushed_result_pause_artifact_invalid",
        )
        try:
            call = execution_calls[index]
            expected_ordinal = getattr(call, "execution_ordinal")
            expected_request_sha256 = getattr(call, "request")["request_sha256"]
            expected_request_bytes = getattr(call, "request_byte_count")
        except Exception:
            _reject("pushed_result_pause_artifact_invalid")
        duration = row["duration_ns"]
        if (
            row["execution_ordinal"] != expected_ordinal
            or row["request_sha256"] != expected_request_sha256
            or row["request_byte_count"] != expected_request_bytes
            or not _is_sha256(row["request_sha256"])
            or type(row["execution_ordinal"]) is not int
            or row["execution_ordinal"] != index + 1
            or type(row["request_byte_count"]) is not int
            or row["request_byte_count"] <= 0
            or type(duration) is not int
            or duration <= 0
        ):
            _reject("pushed_result_pause_artifact_invalid")
        rows.append(row)
        durations.append(duration)
    projected = contract.projected_pilot_ns(durations)
    if (
        item["schema_version"] != PUSHED_RESULT_LATENCY_SCHEMA_VERSION
        or item["selection"]
        != "five_largest_request_bytes_desc_accession_asc"
        or item["remaining_order"]
        != "availability_acceptance_accession_ascending"
        or item["pilot_order_sha256"] != contract.canonical_sha256(rows)
        or item["pilot_count"] != contract.DEVELOPMENT_PILOT_COUNT
        or item["remaining_count"] != contract.DEVELOPMENT_REMAINING_COUNT
        or item["formula"]
        != "sum(pilot_duration_ns)+70*max(pilot_duration_ns)"
        or item["projected_ns"] != projected
        or item["threshold_ns"] != contract.PILOT_PROJECTED_THRESHOLD_NS
        or item["pause_required"] is not True
        or projected <= contract.PILOT_PROJECTED_THRESHOLD_NS
        or item["timing_interval"]
        != (
            "monotonic_ns_after_durable_intent_before_transport_through_"
            "bounded_body_framing_and_close_before_persistence_or_semantic_parse"
        )
    ):
        _reject("pushed_result_pause_artifact_invalid")
    return item


def _pilot_pause_effect_report() -> dict[str, Any]:
    attempted = {
        "yahoo_requests": contract.YAHOO_REQUEST_COUNT,
        "ollama_identity_http_requests": 4,
        "ollama_chat_generations": contract.DEVELOPMENT_PILOT_COUNT,
    }
    request_count = sum(attempted.values())
    return {
        "attempted_external_requests": attempted,
        "completed_effect_counts": contract.build_effect_budgets()["pilot_pause"],
        "request_intent_count": request_count,
        "response_count": request_count,
        "checkpoint_count": request_count + 1,
        "no_retry_repair_pull_fallback_paid_or_trading_effect": True,
    }


def _authenticate_result_continuation_prerequisites(
    *,
    parent: Mapping[str, Any],
    dependencies: PushedResultDependencies,
    store: Any,
    authority: Mapping[str, Any],
    plan: Any,
) -> str:
    """Replay exact private pilot -> public S -> public C at a result commit."""

    try:
        evidence = dependencies.rebuild_pilot_evidence(store=store, plan=plan)
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    if type(evidence) is not tuple or len(evidence) != 2:
        _reject("pushed_result_pause_artifact_invalid")
    pilot_guard = _validate_replayed_pilot_guard(evidence[0])
    latency = _validate_replayed_pilot_latency(evidence[1], plan=plan)
    try:
        plan_manifest = _plain_json_mapping(
            getattr(plan, "manifest"),
            code="pushed_result_pause_artifact_invalid",
        )
        plan_manifest = contract.validate_self_sha256(
            plan_manifest, field="model_plan_sha256"
        )
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    if (
        not _is_sha256(plan_manifest.get("model_plan_sha256"))
        or plan_manifest.get("remaining_order_sha256")
        != authority["request_order"]["remaining_order_sha256"]
    ):
        _reject("pushed_result_pause_artifact_invalid")
    pilot_effect = _pilot_pause_effect_report()
    yahoo_body_bytes = getattr(store.snapshot, "yahoo_body_bytes", None)
    if type(yahoo_body_bytes) is not int or yahoo_body_bytes <= 0:
        _reject("pushed_result_pause_artifact_invalid")
    try:
        expected_pause_raw = dependencies.build_public_pause_artifact(
            authority=authority,
            plan=plan,
            pilot_guard=pilot_guard,
            latency_receipt=latency,
            effect_report={
                **pilot_effect,
                "yahoo_body_bytes": yahoo_body_bytes,
            },
        )
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    expected_pause = _strict_mapping(
        expected_pause_raw,
        _PUSHED_RESULT_PAUSE_FIELDS,
        code="pushed_result_pause_artifact_invalid",
    )
    try:
        expected_pause = contract.validate_self_sha256(
            expected_pause, field="pause_artifact_sha256"
        )
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    durations = [row["duration_ns"] for row in latency["pilot_rows"]]
    if (
        expected_pause["schema_version"] != PUSHED_RESULT_PAUSE_SCHEMA_VERSION
        or expected_pause["status"] != "paused_for_justification"
        or expected_pause["stage"] != contract.DEVELOPMENT_COMMAND
        or expected_pause["attempt_id"] != contract.DEVELOPMENT_ATTEMPT_ID
        or expected_pause["branch"] != contract.BRANCH_NAME
        or expected_pause["preflight_commit"] != parent["preflight_commit"]
        or expected_pause["implementation_commit"]
        != authority["implementation"]["commit"]
        or expected_pause["source_bridge_sha256"]
        != authority["source"]["bridge_sha256"]
        or expected_pause["science_projection_sha256"]
        != contract.SCIENCE_PROJECTION_SHA256
        or expected_pause["model_plan_sha256"]
        != plan_manifest["model_plan_sha256"]
        or expected_pause["pilot_count"] != contract.DEVELOPMENT_PILOT_COUNT
        or expected_pause["pilot_durations_ns"] != durations
        or expected_pause["formula"] != latency["formula"]
        or expected_pause["projected_ns"] != latency["projected_ns"]
        or expected_pause["threshold_ns"] != latency["threshold_ns"]
        or expected_pause["strictly_greater_pause"] is not True
        or expected_pause["pilot_guard_sha256"]
        != pilot_guard["segment_guard_sha256"]
        or expected_pause["latency_receipt_sha256"]
        != latency["latency_receipt_sha256"]
        or expected_pause["remaining_order_sha256"]
        != plan_manifest["remaining_order_sha256"]
        or expected_pause["effect_report"] != pilot_effect
        or expected_pause["model_responses_opened"] is not False
        or expected_pause["market_values_opened"] is not False
        or expected_pause["sixth_generation_attempted"] is not False
        or expected_pause["continuation_requires_fresh_explicit_permission"]
        is not True
        or expected_pause["confirmation_and_final_opened"] is not False
        or expected_pause["privacy_passed"] is not True
    ):
        _reject("pushed_result_pause_artifact_invalid")
    expected_pause_bytes = contract.canonical_json_bytes(expected_pause)
    committed_pause = _strict_mapping(
        parent.get("pause_value"),
        _PUSHED_RESULT_PAUSE_FIELDS,
        code="pushed_result_pause_artifact_invalid",
    )
    try:
        committed_pause = contract.validate_self_sha256(
            committed_pause, field="pause_artifact_sha256"
        )
    except Exception:
        _reject("pushed_result_pause_artifact_invalid")
    if (
        committed_pause != expected_pause
        or parent.get("pause_blob", {}).get("payload") != expected_pause_bytes
    ):
        _reject("pushed_result_pause_artifact_invalid")
    try:
        expected_continuation = dependencies.build_continuation_preregistration(
            expected_pause
        )
    except Exception:
        _reject("pushed_result_continuation_invalid")
    if (
        type(expected_continuation) is not bytes
        or not expected_continuation
        or len(expected_continuation) > MAX_COMPARISON_BYTES
        or parent.get("continuation_blob", {}).get("payload")
        != expected_continuation
    ):
        _reject("pushed_result_continuation_invalid")
    return hashlib.sha256(expected_continuation).hexdigest()


def _classify_result_parent(root: Path, parent: str) -> dict[str, Any]:
    """Reconstruct F or F->S->C strictly from committed Git objects."""

    parent_parent = _git_single_parent(
        root, parent, code="pushed_result_parent_invalid"
    )
    direct = _changed_paths(root, parent_parent, parent)
    if direct == {contract.PREFLIGHT_ARTIFACT_PATH: "A"}:
        return {
            "authorized_parent": parent,
            "authorized_parent_kind": "preflight",
            "preflight_commit": parent,
            "pause_commit": None,
            "continuation_commit": None,
            "continuation_document_sha256": None,
            "pause_blob": None,
            "pause_value": None,
            "continuation_blob": None,
        }
    if direct != {contract.CONTINUATION_PREREGISTRATION_PATH: "A"}:
        _reject("pushed_result_parent_invalid")
    continuation_commit = parent
    pause_commit = parent_parent
    preflight_commit = _git_single_parent(
        root, pause_commit, code="pushed_result_parent_invalid"
    )
    if (
        _changed_paths(root, preflight_commit, pause_commit)
        != {contract.PAUSE_ARTIFACT_PATH: "A"}
        or _changed_paths(root, preflight_commit, continuation_commit)
        != {
            contract.PAUSE_ARTIFACT_PATH: "A",
            contract.CONTINUATION_PREREGISTRATION_PATH: "A",
        }
    ):
        _reject("pushed_result_parent_invalid")
    pause_blob = _git_blob_material(
        root,
        pause_commit,
        contract.PAUSE_ARTIFACT_PATH,
        maximum=MAX_RESULT_ARTIFACT_BYTES,
        code="pushed_result_pause_artifact_invalid",
    )
    pause_value = _parse_canonical_mapping_bytes(
        pause_blob["payload"],
        maximum=MAX_RESULT_ARTIFACT_BYTES,
        code="pushed_result_pause_artifact_invalid",
    )
    continuation_blob = _git_blob_material(
        root,
        continuation_commit,
        contract.CONTINUATION_PREREGISTRATION_PATH,
        maximum=MAX_COMPARISON_BYTES,
        code="pushed_result_continuation_invalid",
    )
    return {
        "authorized_parent": continuation_commit,
        "authorized_parent_kind": "continuation",
        "preflight_commit": preflight_commit,
        "pause_commit": pause_commit,
        "continuation_commit": continuation_commit,
        "continuation_document_sha256": None,
        "pause_blob": pause_blob,
        "pause_value": pause_value,
        "continuation_blob": continuation_blob,
    }


def _authenticate_preflight_revision(
    root: Path,
    *,
    preflight_commit: str,
    direct_head: bool,
    allow_publication_dirty: bool = False,
) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    current_head = _git(root, "rev-parse", "HEAD")
    remote = _git(root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    tree = _git(root, "rev-parse", f"{preflight_commit}^{{tree}}")
    parent = _git(root, "rev-parse", f"{preflight_commit}^")
    parent_tree = _git(root, "rev-parse", f"{parent}^{{tree}}")
    if not all(
        type(item) is str
        for item in (
            branch,
            current_head,
            remote,
            status,
            tree,
            parent,
            parent_tree,
        )
    ):
        _reject("execution_preflight_git_invalid")
    if (
        branch != contract.BRANCH_NAME
        or status != "" and not allow_publication_dirty
        or (direct_head and (current_head != preflight_commit or remote != preflight_commit))
    ):
        _reject("execution_preflight_ancestry_invalid")
    changed = _changed_paths(root, parent, preflight_commit)

    private_manifest, private_payload, completion, _completion_payload = (
        _read_execution_private_state(root)
    )
    repository = validate_repository_snapshot(private_manifest["repository"])
    if (
        repository["ancestry"]["commit"] != parent
        or repository["ancestry"]["tree"] != parent_tree
    ):
        _reject("execution_preflight_implementation_invalid")
    preflight_ancestry = {
        "branch": branch,
        "commit": preflight_commit,
        "tree": tree,
        "parent": parent,
        "implementation_commit": parent,
        "implementation_tree": parent_tree,
        # In descendant reconstruction these are the authenticated historical F
        # identity, not a claim about current HEAD.  The caller separately proves
        # the exact S/C descendant chain and current remote equality.
        "local_head": preflight_commit,
        "remote_head": preflight_commit,
        "changed_paths": changed,
        # A recovery caller proves the only allowed dirty paths separately;
        # this Boolean authenticates the historical F commit itself.
        "clean_worktree": status == "" or allow_publication_dirty,
        "implementation_authenticated": True,
        "private_replay_passed": True,
        "zero_effects_verified": True,
        "preflight_run_count": 1,
    }
    try:
        contract.validate_preflight_ancestry(preflight_ancestry)
    except contract.ContractViolation:
        _reject("execution_preflight_ancestry_invalid")

    public_path = root / Path(contract.PREFLIGHT_ARTIFACT_PATH)
    try:
        public_payload = public_path.read_bytes()
    except OSError:
        _reject("execution_preflight_public_artifact_invalid")
    public_artifact = validate_public_preflight_artifact(
        _parse_artifact_bytes(
            public_payload,
            maximum=1024 * 1024,
            code="execution_preflight_public_artifact_invalid",
        )
    )
    public_literal = hashlib.sha256(public_payload).hexdigest()
    public_blob = _git(
        root,
        "rev-parse",
        f"{preflight_commit}:{contract.PREFLIGHT_ARTIFACT_PATH}",
    )
    public_git_payload = _git(
        root,
        "show",
        f"{preflight_commit}:{contract.PREFLIGHT_ARTIFACT_PATH}",
        binary=True,
    )
    if (
        not _is_sha1(public_blob)
        or type(public_git_payload) is not bytes
        or public_git_payload != public_payload
        or completion["public_artifact_sha256"]
        != public_artifact["public_artifact_sha256"]
        or completion["public_artifact_literal_sha256"] != public_literal
        or public_artifact["private_manifest_sha256"]
        != private_manifest["private_manifest_sha256"]
        or public_artifact["private_manifest_literal_sha256"]
        != hashlib.sha256(private_payload).hexdigest()
        or public_artifact["implementation_commit"] != parent
        or public_artifact["implementation_tree"] != parent_tree
    ):
        _reject("execution_preflight_public_private_mismatch")

    for row in repository["source_inventory"]:
        working_path = root / Path(row["path"])
        try:
            working = working_path.read_bytes()
        except OSError:
            _reject("execution_preflight_source_inventory_invalid")
        blob = _git(root, "show", f"{preflight_commit}:{row['path']}", binary=True)
        blob_sha1 = _git(root, "rev-parse", f"{preflight_commit}:{row['path']}")
        if (
            type(blob) is not bytes
            or type(blob_sha1) is not str
            or working != blob
            or hashlib.sha256(blob).hexdigest() != row["literal_sha256"]
            or blob_sha1 != row["git_blob_sha1"]
            or len(blob) != row["byte_count"]
        ):
            _reject("execution_preflight_source_inventory_invalid")

    forbidden = (
        load_private_contact(root).encode("utf-8"),
        str(root).encode("utf-8"),
        str((root / V38_PRIVATE_ROOT).resolve(strict=False)).encode("utf-8"),
        str((root / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False)).encode(
            "utf-8"
        ),
    )
    _scan_private_bytes(private_payload, forbidden)
    _scan_public_bytes(public_payload, forbidden)

    unsigned = {
        "schema_version": EXECUTION_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": (
            "direct_pushed_preflight" if direct_head else "descendant_reconstruction"
        ),
        "stage": contract.DEVELOPMENT_COMMAND,
        "branch": contract.BRANCH_NAME,
        "preflight_commit": preflight_commit,
        "preflight_tree": tree,
        "implementation_commit": parent,
        "implementation_tree": parent_tree,
        "public_artifact_git_blob_sha1": public_blob,
        "public_artifact_sha256": public_artifact["public_artifact_sha256"],
        "public_artifact_literal_sha256": public_literal,
        "private_manifest_sha256": private_manifest["private_manifest_sha256"],
        "private_manifest_literal_sha256": public_artifact[
            "private_manifest_literal_sha256"
        ],
        "bridge_sha256": public_artifact["aggregates"]["bridge_sha256"],
        "canonical_requests_sha256": public_artifact["aggregates"][
            "canonical_requests_sha256"
        ],
        "pilot_order_sha256": public_artifact["aggregates"][
            "pilot_order_sha256"
        ],
        "remaining_order_sha256": public_artifact["aggregates"][
            "remaining_order_sha256"
        ],
        "production_source_inventory_sha256": public_artifact["aggregates"][
            "production_source_inventory_sha256"
        ],
        "test_source_inventory_sha256": public_artifact["aggregates"][
            "test_source_inventory_sha256"
        ],
        "effect_counts": copy.deepcopy(public_artifact["effect_counts"]),
        "pushed_preflight_gate_passed": True,
        "development_authorized": direct_head,
    }
    return _self_hash(unsigned, "execution_authority_sha256")


def authenticate_execution_preflight(repo_root: Path) -> dict[str, Any]:
    """Authenticate pushed preflight F at exact local/remote HEAD, read-only."""

    root = _canonical_repo_root(repo_root)
    commit = _git(root, "rev-parse", "HEAD")
    parent = _git(root, "rev-parse", "HEAD^")
    if type(commit) is not str or type(parent) is not str:
        _reject("execution_preflight_git_invalid")
    if _changed_paths(root, parent, commit) != {
        contract.PREFLIGHT_ARTIFACT_PATH: "A"
    }:
        # Direct authentication is intentionally valid only while F itself is
        # the pushed HEAD.  A pause or continuation descendant must use the
        # reconstruction path, which also validates its complete ancestry.
        _reject("execution_preflight_ancestry_invalid")
    return _authenticate_preflight_revision(
        root,
        preflight_commit=commit,
        direct_head=True,
    )


def _store_execution_authority(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact stable nine-key authority bound by AttemptStore."""

    return {
        "plan": contract.CONTRACT_MANIFEST_SHA256,
        "attempt": contract.DEVELOPMENT_ATTEMPT_ID,
        "implementation": {
            "commit": receipt["implementation_commit"],
            "tree": receipt["implementation_tree"],
            "production_source_inventory_sha256": receipt[
                "production_source_inventory_sha256"
            ],
            "test_source_inventory_sha256": receipt[
                "test_source_inventory_sha256"
            ],
        },
        "preflight": {
            "commit": receipt["preflight_commit"],
            "tree": receipt["preflight_tree"],
            "public_artifact_sha256": receipt["public_artifact_sha256"],
            "public_artifact_literal_sha256": receipt[
                "public_artifact_literal_sha256"
            ],
            "private_manifest_sha256": receipt["private_manifest_sha256"],
            "private_manifest_literal_sha256": receipt[
                "private_manifest_literal_sha256"
            ],
        },
        "source": {
            "base_commit": contract.BASE_COMMIT,
            "base_tree": contract.BASE_TREE,
            "terminal_internal_sha256": contract.V38_TERMINAL_INTERNAL_SHA256,
            "inventory_sha256": contract.V38_INVENTORY_SHA256,
            "bridge_sha256": receipt["bridge_sha256"],
        },
        "science": {"projection_sha256": contract.SCIENCE_PROJECTION_SHA256},
        "effect_budget": contract.build_effect_budgets(),
        "request_order": {
            "count": contract.DEVELOPMENT_DOCUMENT_COUNT,
            "canonical_requests_sha256": receipt["canonical_requests_sha256"],
            "remaining_order_sha256": receipt["remaining_order_sha256"],
        },
        "pilot_order": {
            "count": contract.DEVELOPMENT_PILOT_COUNT,
            "pilot_order_sha256": receipt["pilot_order_sha256"],
        },
    }


def reconstruct_frozen_execution_authority(repo_root: Path) -> dict[str, Any]:
    """Rebuild exact F authority under an allowed S or C descendant.

    This is read-only and does not validate fresh user permission or authorize a
    continuation effect.  It exists so status/replay can reopen the original
    AttemptStore authority after the preregistered pause commits.
    """

    root = _canonical_repo_root(repo_root)
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    remote = _git(root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if not all(type(item) is str for item in (branch, head, remote, status)):
        _reject("execution_preflight_descendant_invalid")
    if (
        branch != contract.BRANCH_NAME
        or head != remote
        or status != ""
    ):
        _reject("execution_preflight_descendant_invalid")
    parent = _git_single_parent(
        root, head, code="execution_preflight_descendant_invalid"
    )

    direct_delta = _changed_paths(root, parent, head)
    if direct_delta == {contract.PAUSE_ARTIFACT_PATH: "A"}:
        preflight_commit = parent
    elif direct_delta == {contract.CONTINUATION_PREREGISTRATION_PATH: "A"}:
        pause_commit = parent
        preflight_commit = _git_single_parent(
            root, pause_commit, code="execution_preflight_descendant_invalid"
        )
        if _changed_paths(root, preflight_commit, pause_commit) != {
            contract.PAUSE_ARTIFACT_PATH: "A"
        }:
            _reject("execution_preflight_descendant_invalid")
        if _changed_paths(root, preflight_commit, head) != {
            contract.PAUSE_ARTIFACT_PATH: "A",
            contract.CONTINUATION_PREREGISTRATION_PATH: "A",
        }:
            _reject("execution_preflight_descendant_invalid")
    else:
        _reject("execution_preflight_descendant_invalid")

    receipt = _authenticate_preflight_revision(
        root,
        preflight_commit=preflight_commit,
        direct_head=False,
    )
    if receipt["development_authorized"] is not False:
        _reject("execution_preflight_descendant_invalid")
    return _store_execution_authority(receipt)


def load_execution_authority(repo_root: Path) -> dict[str, Any]:
    """Load stable AttemptStore authority at F, S, C, or clean pushed R.

    Reconstructing authority at R is only for safe status/replay access.  It
    does not claim that the separate pushed-result gate passed and cannot
    authorize another development effect.
    """

    root = _canonical_repo_root(repo_root)
    head = _git(root, "rev-parse", "HEAD")
    parent = _git(root, "rev-parse", "HEAD^")
    if type(head) is not str or type(parent) is not str:
        _reject("execution_preflight_git_invalid")
    delta = _changed_paths(root, parent, head)
    if delta == {contract.PREFLIGHT_ARTIFACT_PATH: "A"}:
        return _store_execution_authority(authenticate_execution_preflight(root))
    if delta == {
        contract.RESULT_ARTIFACT_PATH: "A",
        contract.COMPARISON_PATH: "M",
    }:
        result_state = _inspect_result_git(root)
        result_parent = _classify_result_parent(root, result_state["parent"])
        receipt = _authenticate_preflight_revision(
            root,
            preflight_commit=result_parent["preflight_commit"],
            direct_head=False,
        )
        return _store_execution_authority(receipt)
    return reconstruct_frozen_execution_authority(root)


def _publication_path_state(root: Path) -> dict[str, Any]:
    """Inspect only the three idempotent partial-publication paths."""

    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    remote = _git(
        root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}"
    )
    staged = _git(root, "diff", "--cached", "--name-status")
    tracked = _git(root, "diff", "--name-status")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    if not all(
        type(item) is str
        for item in (branch, head, remote, staged, tracked, untracked)
    ):
        _reject("publication_recovery_git_invalid")
    if (
        branch != contract.BRANCH_NAME
        or not _is_sha1(head)
        or head != remote
        or staged != ""
    ):
        _reject("publication_recovery_git_invalid")
    tracked_rows = tracked.splitlines() if tracked else []
    allowed_tracked = {f"M\t{contract.COMPARISON_PATH}"}
    if any(row not in allowed_tracked for row in tracked_rows) or len(tracked_rows) > 1:
        _reject("publication_recovery_dirty_invalid")
    pause_pending_path = f"{contract.PAUSE_ARTIFACT_PATH}.v39-pending"
    result_pending_path = f"{contract.RESULT_ARTIFACT_PATH}.v39-pending"
    comparison_pending_path = f"{contract.COMPARISON_PATH}.v39-pending"
    untracked_paths = set(untracked.splitlines()) if untracked else set()
    if len(untracked_paths) != len(untracked.splitlines()) or not untracked_paths <= {
        contract.PAUSE_ARTIFACT_PATH,
        pause_pending_path,
        contract.RESULT_ARTIFACT_PATH,
        result_pending_path,
        comparison_pending_path,
    }:
        _reject("publication_recovery_dirty_invalid")

    files: dict[str, dict[str, Any] | None] = {}
    for relative in (
        contract.RESULT_ARTIFACT_PATH,
        result_pending_path,
        contract.PAUSE_ARTIFACT_PATH,
        pause_pending_path,
        contract.COMPARISON_PATH,
        comparison_pending_path,
    ):
        path = root / Path(relative)
        try:
            details = path.lstat()
        except FileNotFoundError:
            files[relative] = None
            continue
        except OSError:
            _reject("publication_recovery_dirty_invalid")
        try:
            payload = path.read_bytes()
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            _reject("publication_recovery_dirty_invalid")
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE
            or details.st_nlink not in {0, 1}
            or details.st_size != len(payload)
        ):
            _reject("publication_recovery_dirty_invalid")
        files[relative] = {
            "literal_sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "payload": payload,
        }
    if (
        files[contract.COMPARISON_PATH] is None
        or any(
            (files[path] is not None) != (path in untracked_paths)
            for path in (
                contract.RESULT_ARTIFACT_PATH,
                result_pending_path,
                pause_pending_path,
                comparison_pending_path,
            )
        )
    ):
        _reject("publication_recovery_dirty_invalid")
    return {
        "branch": branch,
        "head": head,
        "remote": remote,
        "tracked_rows": tracked_rows,
        "untracked_paths": sorted(untracked_paths),
        "files": files,
    }


def build_production_publication_recovery_dependencies(
) -> PublicationRecoveryDependencies:
    """Bind terminal-store replay for publication recovery without effects."""

    try:
        from .sec_gemma_lean_science_v39_bridge import (
            authenticate_v38_private_root,
            build_streaming_science_projection,
        )
        from .sec_gemma_lean_science_v39_runner import (
            _pilot_evidence,
            build_attempt_authority,
            build_comparison_update,
            build_effect_report,
            build_model_plan,
            build_preflight_request_commitments,
            build_public_pause_artifact,
            build_public_terminal_artifact,
        )
        from .sec_gemma_lean_science_v39_store import AttemptStore
    except Exception:
        _reject("publication_recovery_dependency_unavailable")

    def rebuild_pause(
        *,
        root: Path,
        store: Any,
        authority: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        contact = load_private_contact(root)
        source = authenticate_v38_private_root(
            root,
            root / V38_PRIVATE_ROOT,
            readable_contact=contact,
        )
        projection = build_streaming_science_projection(source)
        commitments = build_preflight_request_commitments(projection)
        plan = build_model_plan(projection)
        rebuilt = build_attempt_authority(
            execution_authority=authority,
            projection=projection,
            commitments=commitments,
            plan=plan,
        )
        if _plain_json_mapping(
            rebuilt, code="publication_recovery_authority_invalid"
        ) != _plain_json_mapping(
            authority, code="publication_recovery_authority_invalid"
        ):
            _reject("publication_recovery_authority_invalid")
        guard, latency = _pilot_evidence(store, plan=plan)
        effects = build_effect_report(store.snapshot)
        try:
            contract.validate_effect_counts(
                effects["completed_effect_counts"], route="pilot_pause"
            )
        except Exception:
            _reject("publication_recovery_pause_invalid")
        return build_public_pause_artifact(
            authority=authority,
            plan=plan,
            pilot_guard=guard,
            latency_receipt=latency,
            effect_report=effects,
        )

    return PublicationRecoveryDependencies(
        authenticate_preflight_revision=lambda root, commit: (
            _authenticate_preflight_revision(
                root,
                preflight_commit=commit,
                direct_head=False,
                allow_publication_dirty=True,
            )
        ),
        open_store=lambda path, authority: AttemptStore.open(
            path, authority=authority
        ),
        build_public_terminal_artifact=build_public_terminal_artifact,
        build_comparison_update=build_comparison_update,
        rebuild_public_pause_artifact=rebuild_pause,
    )


def load_publication_recovery_authority(
    repo_root: Path,
    *,
    dependencies: PublicationRecoveryDependencies | None = None,
) -> dict[str, Any]:
    """Recover F/C authority only for an exact partial pause/result publication.

    This path can reopen an already terminal-sealed store and compare bytes. It
    returns the same stable nine-key authority used by the attempt; it grants
    no permission to perform or retry any external effect.
    """

    root = _canonical_repo_root(repo_root)
    deps = (
        dependencies
        if dependencies is not None
        else build_production_publication_recovery_dependencies()
    )
    if type(deps) is not PublicationRecoveryDependencies:
        _reject("publication_recovery_dependencies_invalid")
    store: Any | None = None
    try:
        before = _publication_path_state(root)
        parent = _classify_result_parent(root, before["head"])
        if parent["authorized_parent_kind"] not in {"preflight", "continuation"}:
            _reject("publication_recovery_parent_invalid")
        preflight_receipt = deps.authenticate_preflight_revision(
            root, parent["preflight_commit"]
        )
        if (
            not isinstance(preflight_receipt, Mapping)
            or preflight_receipt.get("preflight_commit")
            != parent["preflight_commit"]
            or preflight_receipt.get("pushed_preflight_gate_passed") is not True
            or preflight_receipt.get("development_authorized") is not False
        ):
            _reject("publication_recovery_authority_invalid")
        authority = _store_execution_authority(preflight_receipt)
        store = deps.open_store(
            root / Path(contract.PRIVATE_DEVELOPMENT_NAMESPACE), authority
        )
        snapshot = store.snapshot
        try:
            receipt = store.committed_terminal_evidence()
        except Exception:
            receipt = None
        files = before["files"]
        pause_pending_path = f"{contract.PAUSE_ARTIFACT_PATH}.v39-pending"
        result_pending_path = f"{contract.RESULT_ARTIFACT_PATH}.v39-pending"
        comparison_pending_path = f"{contract.COMPARISON_PATH}.v39-pending"
        committed_comparison = _git_blob_material(
            root,
            before["head"],
            contract.COMPARISON_PATH,
            maximum=MAX_COMPARISON_BYTES,
            code="publication_recovery_comparison_invalid",
        )["payload"]
        if receipt is None:
            pause_builder = deps.rebuild_public_pause_artifact
            if (
                getattr(snapshot, "status", None) != "paused"
                or getattr(snapshot, "paused", None) is not True
                or getattr(snapshot, "continuation_authorized", None) is not False
                or parent["authorized_parent_kind"] != "preflight"
                or pause_builder is None
            ):
                _reject("publication_recovery_pause_invalid")
            expected_pause_raw = pause_builder(
                root=root, store=store, authority=authority
            )
            expected_pause = _plain_json_mapping(
                expected_pause_raw, code="publication_recovery_pause_invalid"
            )
            pause_bytes = contract.canonical_json_bytes(expected_pause)
            for path in (contract.PAUSE_ARTIFACT_PATH, pause_pending_path):
                if files[path] is not None and files[path]["payload"] != pause_bytes:
                    _reject("publication_recovery_pause_invalid")
            if (
                (
                    files[contract.PAUSE_ARTIFACT_PATH] is not None
                    and files[pause_pending_path] is not None
                )
                or
                files[contract.RESULT_ARTIFACT_PATH] is not None
                or files[result_pending_path] is not None
                or files[comparison_pending_path] is not None
                or files[contract.COMPARISON_PATH]["payload"]
                != committed_comparison
                or before["tracked_rows"]
                or (
                    files[contract.PAUSE_ARTIFACT_PATH] is not None
                    and contract.PAUSE_ARTIFACT_PATH
                    not in before["untracked_paths"]
                )
            ):
                _reject("publication_recovery_dirty_invalid")
        else:
            if (
                getattr(snapshot, "status", None)
                not in {"completed", "rejected", "indeterminate"}
                or getattr(snapshot, "journal_head_sha256", None)
                != getattr(receipt, "event_sha256", None)
            ):
                _reject("publication_recovery_terminal_invalid")
            if parent["authorized_parent_kind"] == "continuation":
                if (
                    getattr(snapshot, "continuation_authorized", None) is not True
                    or getattr(snapshot, "continuation_commit", None)
                    != parent["continuation_commit"]
                ):
                    _reject("publication_recovery_parent_invalid")
            elif getattr(snapshot, "continuation_commit", None) is not None:
                _reject("publication_recovery_parent_invalid")
            public = deps.build_public_terminal_artifact(
                authority=authority, terminal_receipt=receipt
            )
            public_value = _plain_json_mapping(
                public, code="publication_recovery_terminal_invalid"
            )
            if (
                public_value.get("invocation_parent") != before["head"]
                or public_value.get("invocation_parent_kind")
                != parent["authorized_parent_kind"]
            ):
                _reject("publication_recovery_parent_invalid")
            result_bytes = contract.canonical_json_bytes(public_value)
            derived_comparison = deps.build_comparison_update(
                committed_comparison, public_value
            )
            if type(derived_comparison) is not bytes:
                _reject("publication_recovery_comparison_invalid")
            for path in (contract.RESULT_ARTIFACT_PATH, result_pending_path):
                if files[path] is not None and files[path]["payload"] != result_bytes:
                    _reject("publication_recovery_result_invalid")
            comparison_file = files[contract.COMPARISON_PATH]
            comparison_pending = files[comparison_pending_path]
            if comparison_file["payload"] not in {
                committed_comparison,
                derived_comparison,
            }:
                _reject("publication_recovery_comparison_invalid")
            if (
                (
                    files[contract.RESULT_ARTIFACT_PATH] is not None
                    and files[result_pending_path] is not None
                )
                or (
                    comparison_pending is not None
                    and comparison_pending["payload"] != derived_comparison
                )
                or (
                    comparison_file["payload"] == derived_comparison
                    and comparison_pending is not None
                )
            ):
                _reject("publication_recovery_pending_invalid")
            comparison_is_dirty = comparison_file["payload"] == derived_comparison
            if (f"M\t{contract.COMPARISON_PATH}" in before["tracked_rows"]) != (
                comparison_is_dirty and derived_comparison != committed_comparison
            ):
                _reject("publication_recovery_dirty_invalid")
            pause_file = files[contract.PAUSE_ARTIFACT_PATH]
            if (
                files[pause_pending_path] is not None
                or parent["authorized_parent_kind"] == "preflight"
                and pause_file is not None
                or parent["authorized_parent_kind"] == "continuation"
                and (
                    pause_file is None
                    or contract.PAUSE_ARTIFACT_PATH in before["untracked_paths"]
                )
            ):
                _reject("publication_recovery_dirty_invalid")
        store.close()
        store = None
        if _publication_path_state(root) != before:
            _reject("publication_recovery_state_changed")
        return authority
    except V39PreflightError:
        raise
    except Exception:
        _reject("publication_recovery_dependency_failed")
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


_COMPLETE_PRIVATE_TERMINAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "attempt_id",
        "invocation_parent",
        "invocation_parent_kind",
        "route",
        "authority_sha256",
        "bridge_manifest",
        "model_plan_manifest",
        "runtime_segment_guards",
        "runtime_aggregate",
        "latency_receipt",
        "stage_slice_sha256",
        "semantic_payload",
        "semantic_payload_sha256",
        "deterministic_payload",
        "deterministic_payload_sha256",
        "science_summary",
        "science_summary_sha256",
        "effect_report",
        "market_values_opened",
        "model_responses_opened",
        "confirmation_and_final_opened",
        "raw_sec_yahoo_or_gemma_response_copied",
        "private_terminal_material_sha256",
    }
)
_FAILURE_PRIVATE_TERMINAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "stage",
        "attempt_id",
        "invocation_parent",
        "invocation_parent_kind",
        "route",
        "authority_sha256",
        "terminal_status",
        "terminal_code",
        "effect_report",
        "journal_head_before_terminal_sha256",
        "closed_segment_generation_counts",
        "market_values_opened",
        "model_responses_opened",
        "source_authenticated",
        "confirmation_and_final_opened",
        "redacted_error_only",
        "private_terminal_material_sha256",
    }
)


def _validate_result_self_hash(value: Any, *, field: str, code: str) -> dict[str, Any]:
    try:
        plain = _plain_json_mapping(value, code=code)
        return contract.validate_self_sha256(plain, field=field)
    except Exception:
        _reject(code)


def _expected_result_route(status: str, parent_kind: str) -> str:
    if parent_kind == "continuation":
        return "paused_resumed"
    if parent_kind != "preflight":
        _reject("pushed_result_parent_invalid")
    return "indeterminate_before_pause" if status == "indeterminate" else "normal"


def _validate_public_result_basics(
    value: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    item = _validate_result_self_hash(
        value,
        field="development_result_sha256",
        code="pushed_result_artifact_invalid",
    )
    terminal_status = item.get("terminal_store_status")
    outcome = item.get("status")
    parent_kind = parent["authorized_parent_kind"]
    if terminal_status not in {"completed", "rejected", "indeterminate"}:
        _reject("pushed_result_artifact_invalid")
    expected_route = _expected_result_route(terminal_status, parent_kind)
    if (
        item.get("stage") != contract.DEVELOPMENT_COMMAND
        or item.get("attempt_id") != contract.DEVELOPMENT_ATTEMPT_ID
        or item.get("branch") != contract.BRANCH_NAME
        or item.get("route") != expected_route
        or item.get("invocation_parent") != parent["authorized_parent"]
        or item.get("invocation_parent_kind") != parent_kind
        or item.get("science_projection_sha256")
        != contract.SCIENCE_PROJECTION_SHA256
        or item.get("contract_manifest_sha256")
        != contract.CONTRACT_MANIFEST_SHA256
        or item.get("real_money_authorized") is not False
        or type(item.get("market_values_opened")) is not bool
        or type(item.get("model_responses_opened")) is not bool
        or item.get("model_responses_opened") is True
        and item.get("market_values_opened") is not True
    ):
        _reject("pushed_result_artifact_invalid")
    terminal_code = item.get("terminal_code")
    if (
        type(terminal_code) is not str
        or re.fullmatch(r"[a-z][a-z0-9_]{0,79}", terminal_code) is None
    ):
        _reject("pushed_result_artifact_invalid")
    if terminal_status == "completed":
        if outcome not in {"passed", "failed_gate"}:
            _reject("pushed_result_artifact_invalid")
    elif outcome != terminal_status:
        _reject("pushed_result_artifact_invalid")
    chronology = item.get("chronology")
    privacy = item.get("privacy")
    limitations = item.get("evidence_limitations")
    if (
        not isinstance(chronology, Mapping)
        or chronology.get("development_only") is not True
        or chronology.get("development_end") != contract.DEVELOPMENT_CORPUS_END
        or chronology.get("confirmation_2019_2023_opened") is not False
        or chronology.get("final_2024_plus_opened") is not False
        or not isinstance(privacy, Mapping)
        or privacy.get("privacy_passed") is not True
        or any(
            privacy.get(key) is not False
            for key in (
                "readable_sec_contact_published",
                "accessions_urls_filenames_offsets_or_bodies_published",
                "canonical_model_requests_published",
                "raw_sec_yahoo_or_gemma_responses_published",
                "realized_transport_metadata_published",
            )
        )
        or privacy.get("redacted_errors_only") is not True
        or type(limitations) is not list
        or "confirmation_and_final_remain_closed" not in limitations
        or "no_real_money_execution_authorized" not in limitations
    ):
        _reject("pushed_result_artifact_invalid")
    if preflight_receipt is not None and (
        item.get("preflight_commit") != preflight_receipt.get("preflight_commit")
        or item.get("preflight_tree") != preflight_receipt.get("preflight_tree")
        or item.get("implementation_commit")
        != preflight_receipt.get("implementation_commit")
        or item.get("implementation_tree")
        != preflight_receipt.get("implementation_tree")
    ):
        _reject("pushed_result_artifact_invalid")
    return item


def _validate_effect_report(
    value: Any,
    *,
    snapshot: Any,
    completed_terminal: bool,
    parent_kind: str,
) -> dict[str, Any]:
    expected_fields = {
        "attempted_external_requests",
        "completed_effect_counts",
        "request_intent_count",
        "response_count",
        "checkpoint_count",
        "yahoo_body_bytes",
        "no_retry_repair_pull_fallback_paid_or_trading_effect",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _reject("pushed_result_effects_invalid")
    item = _plain_json_mapping(value, code="pushed_result_effects_invalid")
    attempted = item["attempted_external_requests"]
    completed = item["completed_effect_counts"]
    expected_attempted = {
        "yahoo_requests",
        "ollama_identity_http_requests",
        "ollama_chat_generations",
    }
    expected_completed = set(contract.build_effect_budgets()["normal_complete"])
    if (
        not isinstance(attempted, Mapping)
        or set(attempted) != expected_attempted
        or not isinstance(completed, Mapping)
        or set(completed) != expected_completed
        or any(type(child) is not int or child < 0 for child in attempted.values())
        or any(type(child) is not int or child < 0 for child in completed.values())
        or item["no_retry_repair_pull_fallback_paid_or_trading_effect"] is not True
    ):
        _reject("pushed_result_effects_invalid")
    if (
        completed["sec_requests"] != 0
        or completed["experiment_family_sec_requests"]
        != contract.EXPERIMENT_FAMILY_SEC_REQUEST_COUNT
        or any(
            completed[key] != 0
            for key in (
                "retries",
                "repairs",
                "pulls",
                "fallbacks",
                "paid_calls",
                "confirmation_final_data_opens",
                "broker_effects",
                "real_money_effects",
            )
        )
    ):
        _reject("pushed_result_effects_invalid")
    snapshot_pairs = {
        "request_intent_count": "request_intent_count",
        "response_count": "response_count",
        "checkpoint_count": "checkpoint_count",
        "yahoo_body_bytes": "yahoo_body_bytes",
    }
    if any(
        type(item[field]) is not int
        or item[field] < 0
        or item[field] != getattr(snapshot, attribute, None)
        for field, attribute in snapshot_pairs.items()
    ):
        _reject("pushed_result_effects_invalid")
    observed_attempted = {
        "yahoo_requests": getattr(snapshot, "yahoo_intent_count", None),
        "ollama_identity_http_requests": getattr(
            snapshot, "identity_intent_count", None
        ),
        "ollama_chat_generations": getattr(snapshot, "gemma_intent_count", None),
    }
    observed_completed = {
        "yahoo_requests": getattr(snapshot, "yahoo_response_count", None),
        "ollama_identity_http_requests": getattr(
            snapshot, "identity_response_count", None
        ),
        "ollama_chat_generations": getattr(snapshot, "gemma_response_count", None),
    }
    if dict(attempted) != observed_attempted or any(
        completed[key] != observed_completed[key] for key in observed_completed
    ):
        _reject("pushed_result_effects_invalid")
    if (
        item["request_intent_count"] != sum(attempted.values())
        or item["response_count"]
        != sum(completed[key] for key in observed_completed)
        or item["checkpoint_count"] > item["response_count"] + 1
        or any(attempted[key] < completed[key] for key in attempted)
    ):
        _reject("pushed_result_effects_invalid")
    if completed_terminal:
        route = (
            "normal_complete"
            if parent_kind == "preflight"
            else "paused_resumed_complete"
        )
        try:
            contract.validate_effect_counts(completed, route=route)
        except Exception:
            _reject("pushed_result_effects_invalid")
        if any(attempted[key] != completed[key] for key in attempted):
            _reject("pushed_result_effects_invalid")
    else:
        maxima = {
            "yahoo_requests": contract.YAHOO_REQUEST_COUNT,
            "ollama_identity_http_requests": 8 if parent_kind == "continuation" else 4,
            "ollama_chat_generations": contract.DEVELOPMENT_DOCUMENT_COUNT,
        }
        if any(
            attempted[key] > maxima[key] or completed[key] > maxima[key]
            for key in maxima
        ):
            _reject("pushed_result_effects_invalid")
    return item


def _validate_private_terminal_material(
    value: Any,
    *,
    receipt: Any,
    snapshot: Any,
    authority: Mapping[str, Any],
    projection_manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    effect_report: Mapping[str, Any],
) -> dict[str, Any]:
    status = getattr(receipt, "status", None)
    completed = status == "completed"
    expected_fields = (
        _COMPLETE_PRIVATE_TERMINAL_FIELDS
        if completed
        else _FAILURE_PRIVATE_TERMINAL_FIELDS
    )
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        _reject("pushed_result_private_terminal_invalid")
    item = _validate_result_self_hash(
        value,
        field="private_terminal_material_sha256",
        code="pushed_result_private_terminal_invalid",
    )
    parent_kind = parent["authorized_parent_kind"]
    if (
        item.get("schema_version")
        != "aapl-sec-gemma-lean-science-v3-9-private-terminal-material-v1"
        or item.get("stage") != contract.DEVELOPMENT_COMMAND
        or item.get("attempt_id") != contract.DEVELOPMENT_ATTEMPT_ID
        or item.get("invocation_parent") != parent["authorized_parent"]
        or item.get("invocation_parent_kind") != parent_kind
        or item.get("route") != _expected_result_route(status, parent_kind)
        or item.get("authority_sha256") != contract.canonical_sha256(authority)
        or item.get("effect_report") != effect_report
        or item.get("confirmation_and_final_opened") is not False
        or type(item.get("market_values_opened")) is not bool
        or type(item.get("model_responses_opened")) is not bool
        or item.get("market_values_opened")
        != getattr(snapshot, "market_values_opened", None)
        or item.get("model_responses_opened")
        != getattr(snapshot, "model_responses_opened", None)
        or item.get("model_responses_opened") is True
        and item.get("market_values_opened") is not True
    ):
        _reject("pushed_result_private_terminal_invalid")
    if completed:
        semantic = item["semantic_payload"]
        deterministic = item["deterministic_payload"]
        summary = item["science_summary"]
        guards = item["runtime_segment_guards"]
        expected_guard_count = 1 if parent_kind == "preflight" else 2
        if (
            item.get("bridge_manifest") != projection_manifest
            or item.get("semantic_payload_sha256")
            != contract.canonical_sha256(semantic)
            or item.get("deterministic_payload_sha256")
            != contract.canonical_sha256(deterministic)
            or item.get("science_summary_sha256")
            != contract.canonical_sha256(summary)
            or type(guards) is not list
            or len(guards) != expected_guard_count
            or item.get("market_values_opened") is not True
            or item.get("model_responses_opened") is not True
            or item.get("raw_sec_yahoo_or_gemma_response_copied") is not False
            or not isinstance(summary, Mapping)
            or not isinstance(summary.get("gate_report"), Mapping)
            or type(summary["gate_report"].get("passed")) is not bool
            or summary.get("no_leverage_proofs_sha256")
            != contract.canonical_sha256(summary.get("no_leverage_proofs"))
        ):
            _reject("pushed_result_private_terminal_invalid")
    elif (
        item.get("terminal_status") != status
        or item.get("terminal_code") != getattr(receipt, "terminal_code", None)
        or item.get("journal_head_before_terminal_sha256")
        != getattr(receipt, "journal_head_before_terminal_sha256", None)
        or type(item.get("closed_segment_generation_counts")) is not list
        or any(
            type(child) is not int or child < 0
            for child in item["closed_segment_generation_counts"]
        )
        or type(item.get("source_authenticated")) is not bool
        or item.get("redacted_error_only") is not True
    ):
        _reject("pushed_result_private_terminal_invalid")
    if not completed:
        completed_counts = effect_report["completed_effect_counts"]
        full_identity = 4 if parent_kind == "preflight" else 8
        all_effects_complete = (
            completed_counts["yahoo_requests"] == contract.YAHOO_REQUEST_COUNT
            and completed_counts["ollama_identity_http_requests"] == full_identity
            and completed_counts["ollama_chat_generations"]
            == contract.DEVELOPMENT_DOCUMENT_COUNT
        )
        if (
            (item["market_values_opened"] or item["model_responses_opened"])
            and not all_effects_complete
            or item["source_authenticated"] is False
            and (
                item["market_values_opened"]
                or item["model_responses_opened"]
                or effect_report["request_intent_count"] != 0
                or effect_report["response_count"] != 0
            )
            or sum(item["closed_segment_generation_counts"])
            > completed_counts["ollama_chat_generations"]
        ):
            _reject("pushed_result_private_terminal_invalid")
    return item


def _comparison_inserted_bytes(before: bytes, after: bytes) -> bytes:
    prefix = 0
    limit = min(len(before), len(after))
    while prefix < limit and before[prefix] == after[prefix]:
        prefix += 1
    suffix = 0
    before_left = len(before) - prefix
    after_left = len(after) - prefix
    while (
        suffix < before_left
        and suffix < after_left
        and before[-(suffix + 1)] == after[-(suffix + 1)]
    ):
        suffix += 1
    end = len(after) - suffix if suffix else len(after)
    return after[prefix:end]


def authenticate_pushed_result(
    repo_root: Path,
    *,
    dependencies: PushedResultDependencies | None = None,
) -> dict[str, Any]:
    """Authenticate pushed R and independently replay its sealed private result.

    The gate is deliberately separate from the one-shot preflight and the
    effectful development worker.  It reads committed Git objects and the
    already-sealed private attempt, returns an ephemeral redacted receipt, and
    never writes an artifact or authorizes confirmation, final, or trading.
    """

    root = _canonical_repo_root(repo_root)
    deps = (
        dependencies
        if dependencies is not None
        else build_production_pushed_result_dependencies()
    )
    if type(deps) is not PushedResultDependencies:
        _reject("pushed_result_dependencies_invalid")
    store: Any | None = None
    try:
        # Complete public/Git authentication happens before contact or private
        # evidence is loaded.
        git_before = _inspect_result_git(root)
        parent = _classify_result_parent(root, git_before["parent"])
        result_payload = git_before["result_blob"]["payload"]
        result = _parse_canonical_mapping_bytes(
            result_payload,
            maximum=MAX_RESULT_ARTIFACT_BYTES,
            code="pushed_result_artifact_invalid",
        )
        result = _validate_public_result_basics(result, parent=parent)
        parent_comparison = _git_blob_material(
            root,
            git_before["parent"],
            contract.COMPARISON_PATH,
            maximum=MAX_COMPARISON_BYTES,
            code="pushed_result_comparison_parent_invalid",
        )["payload"]
        expected_comparison = deps.build_comparison_update(parent_comparison, result)
        if (
            type(expected_comparison) is not bytes
            or expected_comparison != git_before["comparison_blob"]["payload"]
        ):
            _reject("pushed_result_comparison_invalid")

        contact = deps.load_private_contact(root)
        if type(contact) is not str or not contact:
            _reject("pushed_result_private_contact_invalid")
        tokens = _privacy_token_bytes(
            root, lambda path: deps.privacy_tokens(path, contact)
        )
        _scan_public_bytes(result_payload, tokens)
        if parent["authorized_parent_kind"] == "continuation":
            _scan_public_bytes(parent["pause_blob"]["payload"], tokens)
            _scan_public_bytes(parent["continuation_blob"]["payload"], tokens)
        comparison_payload = git_before["comparison_blob"]["payload"]
        if _payload_contains_forbidden_token(comparison_payload, tokens):
            _reject("pushed_result_public_privacy_failed")
        inserted = _comparison_inserted_bytes(parent_comparison, comparison_payload)
        if not inserted:
            _reject("pushed_result_comparison_invalid")
        _scan_public_bytes(inserted, tokens)

        v39_before = _snapshot_serialized_tree(
            root / Path(contract.PRIVATE_NAMESPACE),
            repo_root=root,
            forbidden=tokens,
            code="pushed_result_private_privacy_failed",
        )
        v38_before = _snapshot_serialized_tree(
            root / V38_PRIVATE_ROOT,
            repo_root=root,
            forbidden=None,
            code="pushed_result_v38_inventory_invalid",
        )

        preflight_receipt = deps.authenticate_preflight_revision(
            root, parent["preflight_commit"]
        )
        if (
            not isinstance(preflight_receipt, Mapping)
            or preflight_receipt.get("preflight_commit")
            != parent["preflight_commit"]
            or preflight_receipt.get("pushed_preflight_gate_passed") is not True
            or preflight_receipt.get("development_authorized") is not False
        ):
            _reject("pushed_result_preflight_authority_invalid")
        preflight_receipt = _plain_json_mapping(
            preflight_receipt, code="pushed_result_preflight_authority_invalid"
        )
        result = _validate_public_result_basics(
            result, parent=parent, preflight_receipt=preflight_receipt
        )
        execution_authority = _store_execution_authority(preflight_receipt)

        source = deps.authenticate_source(root, contact)
        projection = deps.build_projection(source)
        projection_manifest_raw = getattr(projection, "manifest", None)
        projection_manifest = validate_bridge_manifest(projection_manifest_raw)
        if (
            projection_manifest["bridge_sha256"]
            != execution_authority["source"]["bridge_sha256"]
            or result.get("source_authority") != execution_authority["source"]
        ):
            _reject("pushed_result_source_authority_invalid")
        context = deps.rebuild_attempt_context(execution_authority, projection)
        if not isinstance(context, Mapping) or set(context) != {"authority", "plan"}:
            _reject("pushed_result_attempt_authority_invalid")
        authority = context["authority"]
        plan = context["plan"]
        if not isinstance(authority, Mapping) or dict(authority) != execution_authority:
            _reject("pushed_result_attempt_authority_invalid")
        authority = _plain_json_mapping(
            authority, code="pushed_result_attempt_authority_invalid"
        )

        store = deps.open_store(
            root / Path(contract.PRIVATE_DEVELOPMENT_NAMESPACE), authority
        )
        snapshot = store.snapshot
        receipt = store.committed_terminal_evidence()
        receipt_evidence = _plain_json_mapping(
            getattr(receipt, "evidence", None),
            code="pushed_result_terminal_seal_invalid",
        )
        terminal_envelope = {
            "schema_version": TERMINAL_EVIDENCE_SCHEMA_VERSION,
            "authority_sha256": contract.canonical_sha256(authority),
            "journal_head_before_terminal_sha256": getattr(
                receipt, "journal_head_before_terminal_sha256", None
            ),
            "status": getattr(receipt, "status", None),
            "terminal_code": getattr(receipt, "terminal_code", None),
            "evidence": receipt_evidence,
        }
        try:
            terminal_envelope_bytes = contract.canonical_json_bytes(
                terminal_envelope
            )
        except Exception:
            _reject("pushed_result_terminal_seal_invalid")
        if (
            getattr(snapshot, "status", None)
            not in {"completed", "rejected", "indeterminate"}
            or getattr(snapshot, "status", None) != getattr(receipt, "status", None)
            or getattr(snapshot, "terminal_code", None)
            != getattr(receipt, "terminal_code", None)
            or getattr(snapshot, "journal_head_sha256", None)
            != getattr(receipt, "event_sha256", None)
            or result.get("terminal_store_status") != getattr(receipt, "status", None)
            or result.get("terminal_code") != getattr(receipt, "terminal_code", None)
            or result.get("private_terminal_evidence_sha256")
            != getattr(receipt, "evidence_sha256", None)
            or result.get("private_terminal_evidence_bytes")
            != getattr(receipt, "evidence_bytes", None)
            or result.get("journal_head_before_terminal_sha256")
            != getattr(receipt, "journal_head_before_terminal_sha256", None)
            or result.get("terminal_event_sha256")
            != getattr(receipt, "event_sha256", None)
            or hashlib.sha256(terminal_envelope_bytes).hexdigest()
            != getattr(receipt, "evidence_sha256", None)
            or len(terminal_envelope_bytes)
            != getattr(receipt, "evidence_bytes", None)
        ):
            _reject("pushed_result_terminal_seal_invalid")

        if parent["authorized_parent_kind"] == "continuation":
            continuation_document_sha256 = (
                _authenticate_result_continuation_prerequisites(
                    parent=parent,
                    dependencies=deps,
                    store=store,
                    authority=authority,
                    plan=plan,
                )
            )
            if (
                getattr(snapshot, "paused", None) is not True
                or getattr(snapshot, "continuation_authorized", None) is not True
                or getattr(snapshot, "continuation_sha256", None)
                != continuation_document_sha256
                or getattr(snapshot, "continuation_commit", None)
                != parent["continuation_commit"]
                or not _is_sha256(
                    getattr(snapshot, "continuation_permission_sha256", None)
                )
            ):
                _reject("pushed_result_continuation_binding_invalid")
        elif (
            getattr(snapshot, "paused", None) is True
            or getattr(snapshot, "continuation_authorized", None) is True
            or getattr(snapshot, "continuation_sha256", None) is not None
            or getattr(snapshot, "continuation_permission_sha256", None) is not None
            or getattr(snapshot, "continuation_commit", None) is not None
        ):
            _reject("pushed_result_continuation_binding_invalid")

        effect_report_raw = deps.build_effect_report(snapshot)
        completed_terminal = getattr(receipt, "status", None) == "completed"
        effect_report = _validate_effect_report(
            effect_report_raw,
            snapshot=snapshot,
            completed_terminal=completed_terminal,
            parent_kind=parent["authorized_parent_kind"],
        )
        public_effect = deps.build_public_effect_report(effect_report)
        if (
            not isinstance(public_effect, Mapping)
            or result.get("effect_report") != dict(public_effect)
        ):
            _reject("pushed_result_effects_invalid")
        private_material = _validate_private_terminal_material(
            receipt_evidence,
            receipt=receipt,
            snapshot=snapshot,
            authority=authority,
            projection_manifest=projection_manifest,
            parent=parent,
            effect_report=effect_report,
        )
        if completed_terminal:
            replayed = deps.replay_completed_terminal(
                store=store,
                authority=authority,
                projection=projection,
                plan=plan,
                effect_report=effect_report,
                invocation_parent=parent["authorized_parent"],
                invocation_parent_kind=parent["authorized_parent_kind"],
            )
        else:
            replayed = deps.replay_failure_terminal(
                snapshot=snapshot,
                receipt=receipt,
                authority=authority,
                effect_report=effect_report,
                invocation_parent=parent["authorized_parent"],
                invocation_parent_kind=parent["authorized_parent_kind"],
                evidence=private_material,
            )
        if not isinstance(replayed, Mapping) or dict(replayed) != private_material:
            _reject("pushed_result_independent_replay_failed")

        expected_public = deps.build_public_terminal_artifact(
            authority=authority, terminal_receipt=receipt
        )
        if not isinstance(expected_public, Mapping) or dict(expected_public) != result:
            _reject("pushed_result_public_replay_failed")
        snapshot_after_replay = store.snapshot
        if snapshot_after_replay != snapshot:
            _reject("pushed_result_store_changed")
        store.close()
        store = None

        # Re-scan both private authorities and Git after all private replay.
        # Equality detects any write or authority race while the gate ran.
        v39_after = _snapshot_serialized_tree(
            root / Path(contract.PRIVATE_NAMESPACE),
            repo_root=root,
            forbidden=tokens,
            code="pushed_result_private_privacy_failed",
        )
        v38_after = _snapshot_serialized_tree(
            root / V38_PRIVATE_ROOT,
            repo_root=root,
            forbidden=None,
            code="pushed_result_v38_inventory_invalid",
        )
        if v39_after != v39_before:
            _reject("pushed_result_private_state_changed")
        if v38_after != v38_before:
            _reject("pushed_result_v38_changed")
        git_after = _inspect_result_git(root)
        if git_after != git_before:
            _reject("pushed_result_git_changed")

        ancestry = {
            "route": result["route"],
            "branch": git_before["branch"],
            "commit": git_before["commit"],
            "tree": git_before["tree"],
            "parent": git_before["parent"],
            "authorized_parent": parent["authorized_parent"],
            "authorized_parent_kind": parent["authorized_parent_kind"],
            "local_head": git_before["commit"],
            "remote_head": git_before["remote"],
            "changed_paths": git_before["changed_paths"],
            "clean_worktree": git_before["status"] == "",
            "terminal_sealed": True,
            "independent_replay_passed": True,
            "privacy_passed": True,
            "v38_unchanged": True,
        }
        try:
            contract.validate_result_ancestry(ancestry)
        except Exception:
            _reject("pushed_result_ancestry_invalid")

        science_summary = private_material.get("science_summary")
        gate_body = {
            "schema_version": PUSHED_RESULT_GATE_SCHEMA_VERSION,
            "status": "passed",
            "development_outcome": result["status"],
            "route": result["route"],
            "branch": git_before["branch"],
            "result_commit": git_before["commit"],
            "result_tree": git_before["tree"],
            "authorized_parent": parent["authorized_parent"],
            "authorized_parent_kind": parent["authorized_parent_kind"],
            "preflight_commit": parent["preflight_commit"],
            "result_git_blob_sha1": git_before["result_blob"]["git_blob_sha1"],
            "result_literal_sha256": git_before["result_blob"]["literal_sha256"],
            "development_result_sha256": result["development_result_sha256"],
            "comparison_git_blob_sha1": git_before["comparison_blob"][
                "git_blob_sha1"
            ],
            "comparison_literal_sha256": git_before["comparison_blob"][
                "literal_sha256"
            ],
            "private_terminal_evidence_sha256": getattr(
                receipt, "evidence_sha256"
            ),
            "terminal_event_sha256": getattr(receipt, "event_sha256"),
            "effect_report_sha256": contract.canonical_sha256(effect_report),
            "science_summary_sha256": (
                contract.canonical_sha256(science_summary)
                if science_summary is not None
                else None
            ),
            "source_bridge_sha256": projection_manifest["bridge_sha256"],
            "private_namespace_inventory_sha256": v39_after[
                "inventory_sha256"
            ],
            "v38_inventory_snapshot_sha256": v38_after["inventory_sha256"],
            "gates": {
                "remote_commit_and_tree_authenticated": True,
                "exact_two_path_delta_authenticated": True,
                "git_blobs_and_canonical_hashes_authenticated": True,
                "private_terminal_seal_replayed": True,
                "effect_counts_replayed": True,
                "deterministic_science_replayed": completed_terminal,
                "failure_material_replayed": not completed_terminal,
                "public_projection_replayed": True,
                "privacy_passed": True,
                "v38_unchanged": True,
                "final_recheck_passed": True,
            },
            "confirmation_preregistration_eligible": result["status"] == "passed",
            "confirmation_execution_authorized": False,
            "final_execution_authorized": False,
            "real_money_authorized": False,
        }
        gate_receipt = {
            **gate_body,
            "pushed_result_gate_sha256": contract.canonical_sha256(gate_body),
        }
        _scan_public_bytes(contract.canonical_json_bytes(gate_receipt), tokens)
        return gate_receipt
    except V39PreflightError:
        raise
    except Exception:
        _reject("pushed_result_dependency_failed")
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


def _run_complete_offline_suite(root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=OFFLINE_TEST_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.SubprocessError):
        _reject("preflight_offline_tests_failed")
    return {
        "schema_version": OFFLINE_TEST_REPORT_SCHEMA_VERSION,
        "suite": "complete_repository_pytest",
        "passed": completed.returncode == 0,
        "exit_code": completed.returncode,
        "command_profile_sha256": OFFLINE_TEST_COMMAND_PROFILE_SHA256,
    }


def _load_private_contact(root: Path) -> str:
    path = root / PRIVATE_CONFIG_PATH
    try:
        details = path.lstat()
        payload = path.read_bytes()
    except OSError:
        _reject("preflight_private_contact_invalid")
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or not payload
        or len(payload) > MAX_PRIVATE_CONFIG_BYTES
    ):
        _reject("preflight_private_contact_invalid")
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
        contact = value["secrets"]["sec_user_agent"]
    except (KeyError, TypeError, ValueError, UnicodeError):
        _reject("preflight_private_contact_invalid")
    if type(contact) is not str or not contact:
        _reject("preflight_private_contact_invalid")
    return contact


def build_production_dependencies() -> PreflightDependencies:
    """Lazily bind local-only production adapters; no work runs here."""

    try:
        from .sec_gemma_lean_science_v39_bridge import (
            authenticate_v38_private_root,
            build_streaming_science_projection,
        )
        from .sec_gemma_lean_science_v39_runner import (
            build_preflight_request_commitments,
        )
    except Exception:
        _reject("preflight_dependency_unavailable")

    cache: dict[str, str] = {}

    def contact(root: Path) -> str:
        if "value" not in cache:
            cache["value"] = _load_private_contact(root)
        return cache["value"]

    def authenticate(root: Path) -> Any:
        return authenticate_v38_private_root(
            root,
            root / V38_PRIVATE_ROOT,
            readable_contact=contact(root),
        )

    def tokens(root: Path) -> tuple[bytes | str, ...]:
        return (
            contact(root),
            str(root),
            str((root / V38_PRIVATE_ROOT).resolve(strict=False)),
            str((root / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False)),
        )

    return PreflightDependencies(
        inspect_repository=inspect_pushed_implementation,
        authenticate_source=authenticate,
        build_projection=build_streaming_science_projection,
        build_request_commitments=build_preflight_request_commitments,
        effect_snapshot=lambda: contract.build_effect_budgets()[
            "zero_effect_preflight"
        ],
        run_offline_tests=_run_complete_offline_suite,
        privacy_tokens=tokens,
    )


def build_production_pushed_result_dependencies() -> PushedResultDependencies:
    """Bind read-only result replay adapters; constructing them performs no work."""

    try:
        from .sec_gemma_lean_science_v39_bridge import (
            authenticate_v38_private_root,
            build_streaming_science_projection,
        )
        from .sec_gemma_lean_science_v39_runner import (
            _open_generation_batch,
            _pilot_evidence,
            _open_stage_slice,
            build_attempt_authority,
            build_comparison_update,
            build_continuation_preregistration,
            build_deterministic_payload,
            build_effect_report,
            build_model_plan,
            build_preflight_request_commitments,
            build_private_terminal_material,
            build_failure_terminal_material,
            build_public_effect_report,
            build_public_pause_artifact,
            build_public_science_summary,
            build_public_terminal_artifact,
            build_semantic_payload,
            evaluate_deterministic_science,
            rebuild_runtime_evidence,
        )
        from .sec_gemma_lean_science_v39_store import AttemptStore
    except Exception:
        _reject("pushed_result_dependency_unavailable")

    def authenticate_preflight(root: Path, commit: str) -> Mapping[str, Any]:
        return _authenticate_preflight_revision(
            root, preflight_commit=commit, direct_head=False
        )

    def authenticate_source(root: Path, contact: str) -> Any:
        return authenticate_v38_private_root(
            root,
            root / V38_PRIVATE_ROOT,
            readable_contact=contact,
        )

    def rebuild_context(
        execution_authority: Mapping[str, Any], projection: Any
    ) -> Mapping[str, Any]:
        commitments = build_preflight_request_commitments(projection)
        plan = build_model_plan(projection)
        authority = build_attempt_authority(
            execution_authority=execution_authority,
            projection=projection,
            commitments=commitments,
            plan=plan,
        )
        return {"authority": authority, "plan": plan}

    def replay_completed(**kwargs: Any) -> Mapping[str, Any]:
        store = kwargs["store"]
        authority = kwargs["authority"]
        projection = kwargs["projection"]
        plan = kwargs["plan"]
        effect_report = kwargs["effect_report"]
        guards, aggregate, latency, guard_by_request = rebuild_runtime_evidence(
            store, plan=plan
        )
        stage_slice = _open_stage_slice(store, plan=plan)
        sealed = _open_generation_batch(store, plan=plan)
        semantic = build_semantic_payload(
            plan=plan,
            sealed_calls_by_request_sha256=sealed,
            segment_guard_by_request_sha256=guard_by_request,
            runtime_aggregate=aggregate,
            latency_receipt=latency,
        )
        deterministic = build_deterministic_payload(
            stage_slice=stage_slice, semantic_payload=semantic
        )
        evaluated = evaluate_deterministic_science(
            semantic_payload=semantic,
            deterministic_payload=deterministic,
        )
        if not isinstance(evaluated, Mapping) or not isinstance(
            evaluated.get("evaluation"), Mapping
        ):
            _reject("pushed_result_deterministic_replay_failed")
        evaluation = dict(evaluated["evaluation"])
        public_summary = build_public_science_summary(evaluation)
        material = build_private_terminal_material(
            authority=authority,
            projection=projection,
            plan=plan,
            runtime_guards=guards,
            runtime_aggregate=aggregate,
            latency_receipt=latency,
            stage_slice=stage_slice,
            semantic_payload=semantic,
            deterministic_payload=deterministic,
            evaluation=evaluation,
            effect_report=effect_report,
            invocation_parent=kwargs["invocation_parent"],
            invocation_parent_kind=kwargs["invocation_parent_kind"],
        )
        if material.get("science_summary") != public_summary:
            _reject("pushed_result_science_summary_replay_failed")
        return material

    def replay_failure(**kwargs: Any) -> Mapping[str, Any]:
        snapshot = kwargs["snapshot"]
        receipt = kwargs["receipt"]
        evidence = kwargs["evidence"]
        try:
            before_terminal = replace(
                snapshot,
                journal_head_sha256=receipt.journal_head_before_terminal_sha256,
            )
        except Exception:
            _reject("pushed_result_failure_replay_failed")
        return build_failure_terminal_material(
            authority=kwargs["authority"],
            terminal_code=receipt.terminal_code,
            terminal_status=receipt.status,
            effect_report=kwargs["effect_report"],
            snapshot=before_terminal,
            invocation_parent=kwargs["invocation_parent"],
            invocation_parent_kind=kwargs["invocation_parent_kind"],
            market_values_opened=evidence["market_values_opened"],
            model_responses_opened=evidence["model_responses_opened"],
            source_authenticated=evidence["source_authenticated"],
        )

    def tokens(root: Path, contact: str) -> tuple[str, ...]:
        v38 = str((root / V38_PRIVATE_ROOT).resolve(strict=False))
        v39 = str((root / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False))
        return (
            contact,
            str(root),
            str(root).replace("\\", "/"),
            v38,
            v38.replace("\\", "/"),
            v39,
            v39.replace("\\", "/"),
        )

    return PushedResultDependencies(
        authenticate_preflight_revision=authenticate_preflight,
        load_private_contact=load_private_contact,
        authenticate_source=authenticate_source,
        build_projection=build_streaming_science_projection,
        rebuild_attempt_context=rebuild_context,
        open_store=lambda path, authority: AttemptStore.open(
            path, authority=authority
        ),
        build_effect_report=build_effect_report,
        build_public_effect_report=build_public_effect_report,
        rebuild_pilot_evidence=lambda **kwargs: _pilot_evidence(
            kwargs["store"], plan=kwargs["plan"]
        ),
        build_public_pause_artifact=build_public_pause_artifact,
        build_continuation_preregistration=build_continuation_preregistration,
        replay_completed_terminal=replay_completed,
        replay_failure_terminal=replay_failure,
        build_public_terminal_artifact=build_public_terminal_artifact,
        build_comparison_update=build_comparison_update,
        privacy_tokens=tokens,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AAPL SEC/Gemma v3.9 zero-effect preflight"
    )
    parser.add_argument("repo_root", nargs="?", default=".")
    parser.add_argument(
        "--pushed-result",
        action="store_true",
        help="run only the separate read-only pushed-result authentication gate",
    )
    arguments = parser.parse_args(argv)
    try:
        result = (
            authenticate_pushed_result(Path(arguments.repo_root))
            if arguments.pushed_result
            else run_preflight(Path(arguments.repo_root))
        )
    except V39PreflightError as exc:
        print(json.dumps({"status": "failed", "code": exc.code}, sort_keys=True))
        return 1
    output = {"status": result["status"]}
    if arguments.pushed_result:
        output["pushed_result_gate_sha256"] = result[
            "pushed_result_gate_sha256"
        ]
    else:
        output["public_artifact_sha256"] = result["public_artifact_sha256"]
    print(json.dumps(output, sort_keys=True))
    return 0


__all__ = [
    "EXECUTION_AUTHORITY_SCHEMA_VERSION",
    "OFFLINE_TEST_REPORT_SCHEMA_VERSION",
    "OFFLINE_TEST_COMMAND_PROFILE_SHA256",
    "OFFLINE_TEST_TIMEOUT_SECONDS",
    "PREFLIGHT_ATTEMPT_ID",
    "PREFLIGHT_SCHEMA_VERSION",
    "PUSHED_RESULT_GATE_SCHEMA_VERSION",
    "PRIVATE_MANIFEST_SCHEMA_VERSION",
    "PUBLIC_ARTIFACT_SCHEMA_VERSION",
    "PUBLIC_FAILED_PREFLIGHT_SCHEMA_VERSION",
    "PreflightDependencies",
    "PublicationRecoveryDependencies",
    "PushedResultDependencies",
    "REQUEST_COMMITMENTS_SCHEMA_VERSION",
    "V39PreflightError",
    "authenticate_execution_preflight",
    "authenticate_pushed_result",
    "build_production_dependencies",
    "build_production_publication_recovery_dependencies",
    "build_production_pushed_result_dependencies",
    "inspect_pushed_implementation",
    "load_execution_authority",
    "load_private_contact",
    "load_publication_recovery_authority",
    "main",
    "run_preflight",
    "reconstruct_frozen_execution_authority",
    "validate_bridge_manifest",
    "validate_offline_test_report",
    "validate_private_preflight_manifest",
    "validate_public_preflight_artifact",
    "validate_public_failed_preflight_artifact",
    "validate_repository_snapshot",
    "validate_request_commitments",
]


if __name__ == "__main__":
    raise SystemExit(main())
