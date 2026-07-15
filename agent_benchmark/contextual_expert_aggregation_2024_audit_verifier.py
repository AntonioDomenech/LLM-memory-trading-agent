"""Read-only independent verifier for the one-shot calendar-2024 audit.

The verifier accepts only the frozen private or final location.  It validates
the flat byte bundle before parsing semantics, then independently regenerates
the reproducible payloads from the sealed parent, bounded input, checkpoint,
ledgers, episode/XOR evidence, and pure evaluation implementation.  Runtime
samples are the sole process-local input accepted by regeneration.

This module performs no network, news, LLM, API, external-model, broker, or
write operation.  In particular, verification never promotes, repairs,
quarantines, removes, or rewrites evidence.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from . import contextual_expert_aggregation_2024_audit_artifacts as _artifacts
from . import contextual_expert_aggregation_2024_audit_computation as _computation
from . import contextual_expert_aggregation_2024_audit_control as _control
from . import contextual_expert_aggregation_2024_audit_evaluation as _evaluation
from . import contextual_expert_aggregation_2024_audit_input as _input
from . import contextual_expert_aggregation_2024_audit_parent as _parent


VERIFIER_ID = _artifacts.VERIFIER_ID
EXPECTED_ARTIFACT_SCHEMA_REGISTRY_SHA256 = (
    "sha256:108fd5696f45b196d6529bf74e931ffbbbd7d1eed534477a2548de4d29857f62"
)
VERIFIER_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_2024_audit_verifier.py"
)
VERIFIER_LIMIT_SECONDS = 1_800.0
PRIVATE_MODE = "private"
STANDALONE_MODE = "standalone"
_MODES = frozenset({PRIVATE_MODE, STANDALONE_MODE})
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

_RUNTIME_FIELDS = frozenset(
    {
        "runtime_evidence_schema_version",
        "contract_version",
        "run_id",
        "stage",
        "clock_name",
        "command_start_monotonic",
        "stage_deadline_seconds",
        "finalization_reserve_seconds",
        "samples",
        "stage_internal_deadline_pass",
        "final_commit_sample_location",
        "network_access",
        "news_access",
        "llm_calls",
        "api_calls",
        "external_model_calls",
        "external_cost_usd",
        "runtime_evidence_sha256",
    }
)
_RUNTIME_SAMPLE_ORDER = ("preseal", "post_private_verify", "prepromotion")

_LOCK_FIELDS = frozenset(
    {
        "lock_schema_version",
        "contract_version",
        "run_id",
        "stage",
        "attempt_number",
        "one_run_no_retry",
        "persistent_after_success_or_failure",
        "expected_branch",
        "preregistration_commit",
        "git_commit",
        "git_identity",
        "dependency_identity_sha256",
        "artifact_schema_registry_sha256",
        "parent_verification",
        "input_identity",
        "receipt_identity",
        "prelock_control_inventory",
        "output_paths",
        "created_before_receipt_or_input_worktree_read",
        "created_before_any_2024_market_value_release",
        "created_at_utc",
    }
)

_NONCOPY_JSON_FILENAMES = frozenset(
    {
        "parent_verification.json",
        "prefix_continuity_proof.json",
        "source_bundle_provenance.json",
        "known_result_isolation_evidence.json",
        "audit_continuation_checkpoint_through_2024.json",
        "audit_pending_lessons.json",
        "audit_replay_diagnostics.json",
        "audit_episodes__base_5bps.json",
        "audit_episodes__stress_10bps.json",
        "audit_xor__base_5bps.json",
        "audit_xor__stress_10bps.json",
        "audit_metrics.json",
        "audit_gate_report.json",
        "audit_integrity_evidence.json",
        "audit_runtime_cost_evidence.json",
        "report.json",
    }
)


class ContextualExpertAggregation2024AuditVerificationError(RuntimeError):
    """Raised when audit evidence cannot be independently verified."""


def _fail(message: str) -> ContextualExpertAggregation2024AuditVerificationError:
    return ContextualExpertAggregation2024AuditVerificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise _fail("raw hash input must be exact bytes")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    # Artifact canonicalization deliberately excludes the terminal LF from
    # every omitted-field self-hash domain.
    return _sha256_bytes(_artifacts.canonical_json_line_bytes(value)[:-1])


def _strict_mapping(
    value: Any, expected: set[str] | frozenset[str], *, field_name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise _fail(f"{field_name} has the wrong exact field inventory")
    return value


def _finite_float(value: Any, *, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise _fail(f"{field_name} must be a canonical finite float")
    return value


def _exact_int(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{field_name} must be an exact integer >= {minimum}")
    return value


def _exact_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise _fail(f"{field_name} must be an exact boolean")
    return value


@dataclass
class VerificationClock:
    """Strict monotonic verifier clock with named, inspectable samples."""

    started_at: float
    now: Callable[[], float] = field(default=time.monotonic, repr=False)
    limit_seconds: float = VERIFIER_LIMIT_SECONDS
    samples: dict[str, float] = field(default_factory=dict)

    @classmethod
    def start(
        cls, *, now: Callable[[], float] = time.monotonic
    ) -> "VerificationClock":
        return cls(started_at=float(now()), now=now)

    def __post_init__(self) -> None:
        if (
            type(self.started_at) is not float
            or not math.isfinite(self.started_at)
            or self.started_at < 0.0
            or type(self.limit_seconds) is not float
            or self.limit_seconds != VERIFIER_LIMIT_SECONDS
        ):
            raise _fail("verifier clock identity or strict limit changed")

    def check(self, phase: str) -> float:
        if type(phase) is not str or not phase or phase in self.samples:
            raise _fail("verifier clock phase is empty or repeated")
        current = self.now()
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            raise _fail("verifier monotonic clock returned a non-number")
        elapsed = float(current) - self.started_at
        if not math.isfinite(elapsed) or not 0.0 <= elapsed < self.limit_seconds:
            raise _fail(f"verifier strict deadline failed at {phase}")
        if self.samples and elapsed < next(reversed(self.samples.values())):
            raise _fail("verifier monotonic samples moved backwards")
        self.samples[phase] = elapsed
        return elapsed


def _clock_check(clock: Any, phase: str) -> float:
    try:
        if hasattr(clock, "check"):
            value = clock.check(phase)
        elif callable(clock):
            value = clock()
        else:
            raise _fail("verifier clock lacks the required check method")
    except ContextualExpertAggregation2024AuditVerificationError:
        raise
    except Exception as exc:
        raise _fail(f"verifier deadline check failed at {phase}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail("verifier deadline sample is not numeric")
    elapsed = float(value)
    if not math.isfinite(elapsed) or not 0.0 <= elapsed < VERIFIER_LIMIT_SECONDS:
        raise _fail(f"verifier strict deadline failed at {phase}")
    return elapsed


def _runtime_payload(
    value: Any, *, allow_provisional_runtime: bool = False
) -> dict[str, Any]:
    runtime = _strict_mapping(value, _RUNTIME_FIELDS, field_name="runtime evidence")
    if (
        type(runtime["runtime_evidence_schema_version"]) is not int
        or runtime["runtime_evidence_schema_version"] != 1
        or runtime["contract_version"] != _artifacts.CONTRACT_VERSION
        or runtime["run_id"] != _artifacts.AUDIT_RUN_ID
        or runtime["stage"] != _artifacts.AUDIT_STAGE
        or type(runtime["clock_name"]) is not str
        or not runtime["clock_name"]
        or runtime["stage_deadline_seconds"] != VERIFIER_LIMIT_SECONDS
        or type(runtime["stage_deadline_seconds"]) is not float
        or runtime["finalization_reserve_seconds"]
        != _artifacts.FINALIZATION_RESERVE_SECONDS
        or type(runtime["finalization_reserve_seconds"]) is not float
        or type(runtime["final_commit_sample_location"]) is not str
        or not runtime["final_commit_sample_location"]
    ):
        raise _fail("runtime evidence frozen identity changed")
    _finite_float(
        runtime["command_start_monotonic"],
        field_name="runtime.command_start_monotonic",
    )
    samples = _strict_mapping(
        runtime["samples"],
        frozenset(_RUNTIME_SAMPLE_ORDER),
        field_name="runtime.samples",
    )
    normalized_samples: dict[str, float | None] = {}
    seen_provisional = False
    previous = -math.inf
    for name in _RUNTIME_SAMPLE_ORDER:
        raw = samples[name]
        if raw is None and allow_provisional_runtime and name != "preseal":
            normalized_samples[name] = None
            seen_provisional = True
            continue
        if seen_provisional:
            raise _fail("provisional runtime samples are not a suffix of nulls")
        selected = _finite_float(raw, field_name=f"runtime.samples.{name}")
        if selected < 0.0 or (previous != -math.inf and selected <= previous):
            raise _fail("runtime samples are negative or not strictly increasing")
        previous = selected
        normalized_samples[name] = selected
    if normalized_samples["preseal"] is None:
        raise _fail("runtime preseal sample may never be provisional")
    prepromotion = normalized_samples["prepromotion"]
    pass_value = _exact_bool(
        runtime["stage_internal_deadline_pass"],
        field_name="runtime.stage_internal_deadline_pass",
    )
    if allow_provisional_runtime and prepromotion is None:
        if (
            normalized_samples["post_private_verify"] is not None
            or pass_value is not False
        ):
            raise _fail("provisional runtime may not claim the final stage deadline")
    elif (
        prepromotion is None
        or not prepromotion
        < VERIFIER_LIMIT_SECONDS - _artifacts.FINALIZATION_RESERVE_SECONDS
        or pass_value is not True
    ):
        raise _fail("runtime evidence violates the strict prepromotion reserve")
    for name in ("network_access", "news_access"):
        if _exact_bool(runtime[name], field_name=f"runtime.{name}") is not False:
            raise _fail("runtime evidence records forbidden external access")
    for name in ("llm_calls", "api_calls", "external_model_calls"):
        if _exact_int(runtime[name], field_name=f"runtime.{name}") != 0:
            raise _fail("runtime evidence records forbidden model or API use")
    if (
        _finite_float(
            runtime["external_cost_usd"], field_name="runtime.external_cost_usd"
        )
        != 0.0
    ):
        raise _fail("runtime evidence records nonzero external cost")
    recorded = runtime["runtime_evidence_sha256"]
    if type(recorded) is not str or _SHA256_RE.fullmatch(recorded) is None:
        raise _fail("runtime evidence self-hash is malformed")
    unsigned = {
        name: runtime[name]
        for name in runtime
        if name != "runtime_evidence_sha256"
    }
    if recorded != _sha256_json(unsigned):
        raise _fail("runtime evidence omitted-field self-hash is invalid")
    return dict(runtime)


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _ordinary(path: Path, *, directory: bool, field_name: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise _fail(f"{field_name} is missing or unreadable") from exc
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    reparse = bool(getattr(info, "st_file_attributes", 0) & 0x0400)
    if not expected or stat.S_ISLNK(info.st_mode) or reparse:
        raise _fail(f"{field_name} is not an ordinary non-reparse path")
    # A bundle entry may never be a mount point.  The repository root itself
    # can legitimately sit on a mounted volume and is handled separately.
    if path.parent != path and path.is_mount():
        raise _fail(f"{field_name} is an unauthorized mount point")


def _read_ordinary(path: Path, *, field_name: str) -> bytes:
    _ordinary(path, directory=False, field_name=field_name)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise _fail(f"{field_name} changed type while opening")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    except ContextualExpertAggregation2024AuditVerificationError:
        raise
    except OSError as exc:
        raise _fail(f"{field_name} became unreadable") from exc
    _ordinary(path, directory=False, field_name=field_name)
    return b"".join(chunks)


def _directory_entries(
    path: Path, expected: Mapping[str, bool], *, field_name: str
) -> None:
    _ordinary(path, directory=True, field_name=field_name)
    try:
        entries = {entry.name: Path(entry.path) for entry in os.scandir(path)}
    except OSError as exc:
        raise _fail(f"{field_name} inventory is unreadable") from exc
    if set(entries) != set(expected):
        raise _fail(f"{field_name} has the wrong exact inventory")
    for name, is_directory in expected.items():
        _ordinary(
            entries[name],
            directory=is_directory,
            field_name=f"{field_name}.{name}",
        )


def _bundle_location(
    directory: Path, *, repo_root: Path, mode: str
) -> tuple[Path, Path]:
    if mode not in _MODES:
        raise _fail("verifier mode is not private or standalone")
    root_input = Path(repo_root)
    bundle_input = Path(directory)
    if ".." in root_input.parts or ".." in bundle_input.parts:
        raise _fail("verifier path contains lexical parent traversal")
    root = _lexical(root_input)
    bundle = _lexical(bundle_input)
    _ordinary(root, directory=True, field_name="repository root")
    expected = _lexical(
        root
        / (
            _artifacts.PRIVATE_DIRECTORY
            if mode == PRIVATE_MODE
            else _artifacts.OUTPUT_DIRECTORY
        )
    )
    if bundle != expected:
        raise _fail("verifier received an unauthorized bundle location")
    try:
        bundle.relative_to(root)
    except ValueError as exc:
        raise _fail("verifier bundle escapes the repository root") from exc
    current = bundle
    while current != root:
        _ordinary(current, directory=True, field_name="bundle path component")
        current = current.parent
    return root, bundle


def _verify_control_layout(root: Path, *, mode: str, bundle: Path) -> None:
    control = root / _artifacts.CONTROL_ROOT
    authorized = root / _input.INPUT_PATH.parent
    runs = root / _artifacts.OUTPUT_PARENT
    root_expected = {
        "authorized_inputs": True,
        _input.RECEIPT_PATH.name: False,
        _artifacts.ATTEMPT_LOCK_FILENAME: False,
        _artifacts.OUTPUT_PARENT.name: True,
    }
    if mode == STANDALONE_MODE:
        root_expected[_artifacts.SUCCESS_MARKER_FILENAME] = False
    _directory_entries(control, root_expected, field_name="audit control root")
    _directory_entries(
        authorized,
        {_input.INPUT_PATH.name: False},
        field_name="authorized input directory",
    )
    _directory_entries(
        runs,
        {
            (
                _artifacts.PRIVATE_DIRECTORY.name
                if mode == PRIVATE_MODE
                else _artifacts.AUDIT_RUN_ID
            ): True
        },
        field_name="audit runs directory",
    )
    if _lexical(bundle.parent) != _lexical(runs):
        raise _fail("bundle is not the sole direct child of the runs directory")
    for forbidden in (
        root / _artifacts.PENDING_SUCCESS_MARKER_PATH,
        root / _artifacts.FAILED_DIRECTORY,
    ):
        if os.path.lexists(forbidden):
            raise _fail("pending or failed audit evidence exists during verification")
    if mode == PRIVATE_MODE and os.path.lexists(
        root / _artifacts.SUCCESS_MARKER_PATH
    ):
        raise _fail("private verification may not observe a success marker")


@dataclass(frozen=True)
class VerifiedAuditByteBundle:
    directory: Path
    files: Mapping[str, bytes]
    manifest: Mapping[str, Any]
    checksums: Mapping[str, str]

    @property
    def manifest_bytes(self) -> bytes:
        return self.files["stage_manifest.json"]

    @property
    def checksums_bytes(self) -> bytes:
        return self.files["checksums.json"]


def _read_bundle_bytes(
    directory: Path, *, expected_manifest_sha256: str | None
) -> VerifiedAuditByteBundle:
    _directory_entries(
        directory,
        {name: False for name in _artifacts.BUNDLE_FILE_ORDER},
        field_name="audit bundle",
    )
    files = {
        name: _read_ordinary(directory / name, field_name=f"bundle.{name}")
        for name in _artifacts.BUNDLE_FILE_ORDER
    }
    try:
        manifest = _artifacts.parse_stage_manifest_bytes(
            files["stage_manifest.json"]
        )
        checksums = _artifacts.parse_checksums_bytes(files["checksums.json"])
    except _artifacts.ContextualExpertAggregation2024AuditArtifactError as exc:
        raise _fail("manifest or checksum artifact is invalid") from exc
    if (
        expected_manifest_sha256 is not None
        and manifest["manifest_sha256"] != expected_manifest_sha256
    ):
        raise _fail("manifest self-hash differs from the independently bound value")
    observed_payload_hashes = {
        name: _sha256_bytes(files[name]) for name in _artifacts.PAYLOAD_FILE_ORDER
    }
    if dict(manifest["payload_sha256"]) != observed_payload_hashes:
        raise _fail("manifest payload hashes differ from the raw bundle bytes")
    expected_checksums = {
        **observed_payload_hashes,
        "stage_manifest.json": _sha256_bytes(files["stage_manifest.json"]),
    }
    if dict(checksums) != expected_checksums:
        raise _fail("checksum map differs from the exact 42-file hash domain")
    if (
        manifest["receipt_file_sha256"]
        != observed_payload_hashes["input_snapshot_receipt.json"]
        or manifest["input_raw_sha256"]
        != observed_payload_hashes["audit_prices_through_2024.csv"]
        or manifest["attempt_lock_file_sha256"]
        != observed_payload_hashes[_artifacts.ATTEMPT_LOCK_FILENAME]
        or manifest["runtime_cost_evidence_file_sha256"]
        != observed_payload_hashes["audit_runtime_cost_evidence.json"]
        or manifest["gate_report_file_sha256"]
        != observed_payload_hashes["audit_gate_report.json"]
        or manifest["parent_manifest_file_sha256"]
        != observed_payload_hashes["parent_stage_manifest.json"]
    ):
        raise _fail("manifest named raw-file hash domains do not match payload bytes")
    if files[".gitattributes"] != _artifacts.GIT_ATTRIBUTES_BYTES:
        raise _fail("bundle .gitattributes bytes changed")
    return VerifiedAuditByteBundle(
        directory=directory,
        files=files,
        manifest=manifest,
        checksums=checksums,
    )


def _validate_schema(value: Any, schema: Any, *, field_name: str) -> None:
    """Validate the small literal schema DSL frozen in the artifact registry."""

    if type(schema) is str:
        if schema.startswith("PENDING_"):
            raise _fail("artifact deep schema registry is still provisional")
        if schema == "string":
            if type(value) is not str:
                raise _fail(f"{field_name} must be a string")
            return
        if schema == "bool":
            _exact_bool(value, field_name=field_name)
            return
        if schema == "int":
            if type(value) is not int:
                raise _fail(f"{field_name} must be an exact integer")
            return
        if schema == "float":
            _finite_float(value, field_name=field_name)
            return
        if schema == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _fail(f"{field_name} must be a finite number")
            if not math.isfinite(float(value)):
                raise _fail(f"{field_name} must be finite")
            return
        if schema == "sha256":
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise _fail(f"{field_name} must be a lowercase SHA-256")
            return
        if schema == "git_commit":
            if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
                raise _fail(f"{field_name} must be a full lowercase Git commit")
            return
        if schema == "iso_date":
            if type(value) is not str:
                raise _fail(f"{field_name} must be an ISO date")
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise _fail(f"{field_name} must be an ISO date") from exc
            if parsed.isoformat() != value:
                raise _fail(f"{field_name} is not a canonical ISO date")
            return
        if schema == "null":
            if value is not None:
                raise _fail(f"{field_name} must be null")
            return
        if schema.startswith("list[") and schema.endswith("]"):
            if type(value) is not list:
                raise _fail(f"{field_name} must be a list")
            item_schema = schema[5:-1]
            for position, item in enumerate(value):
                _validate_schema(
                    item, item_schema, field_name=f"{field_name}[{position}]"
                )
            return
        if schema.startswith("map[string,") and schema.endswith("]"):
            if not isinstance(value, Mapping) or any(
                type(key) is not str for key in value
            ):
                raise _fail(f"{field_name} must be a string-keyed map")
            item_schema = schema[len("map[string,") : -1]
            for key, item in value.items():
                _validate_schema(item, item_schema, field_name=f"{field_name}.{key}")
            return
        raise _fail(f"{field_name} uses an unsupported frozen schema primitive")

    if isinstance(schema, Mapping):
        schema_type = schema.get("type")
        if schema_type == "object":
            fields = schema.get("fields")
            if not isinstance(fields, Mapping):
                raise _fail(f"{field_name} object schema is malformed")
            mapping = _strict_mapping(
                value, set(fields), field_name=field_name
            )
            for name, child_schema in fields.items():
                _validate_schema(
                    mapping[name], child_schema, field_name=f"{field_name}.{name}"
                )
            self_hash_field = schema.get("self_hash_field")
            if self_hash_field is not None:
                if type(self_hash_field) is not str or self_hash_field not in mapping:
                    raise _fail(f"{field_name} self-hash schema is malformed")
                unsigned = {
                    name: mapping[name]
                    for name in mapping
                    if name != self_hash_field
                }
                if mapping[self_hash_field] != _sha256_json(unsigned):
                    raise _fail(f"{field_name} omitted-field self-hash is invalid")
            return
        if schema_type == "fixed_map":
            keys = schema.get("keys")
            if not isinstance(keys, (list, tuple)) or any(
                type(key) is not str for key in keys
            ):
                raise _fail(f"{field_name} fixed-map schema is malformed")
            mapping = _strict_mapping(value, set(keys), field_name=field_name)
            child_schema = schema.get("values")
            for key in keys:
                _validate_schema(
                    mapping[key], child_schema, field_name=f"{field_name}.{key}"
                )
            return
        if schema_type == "list":
            if type(value) is not list:
                raise _fail(f"{field_name} must be a list")
            if "length" in schema and len(value) != schema["length"]:
                raise _fail(f"{field_name} list length changed")
            for position, item in enumerate(value):
                _validate_schema(
                    item,
                    schema.get("items"),
                    field_name=f"{field_name}[{position}]",
                )
            return
        if schema_type == "map":
            if not isinstance(value, Mapping) or any(
                type(key) is not str for key in value
            ):
                raise _fail(f"{field_name} must be a string-keyed map")
            for key, item in value.items():
                _validate_schema(
                    item, schema.get("values"), field_name=f"{field_name}.{key}"
                )
            return
        if schema_type == "enum":
            if value not in schema.get("values", []):
                raise _fail(f"{field_name} is outside its frozen enum")
            return
        if schema_type == "literal":
            if type(value) is not type(schema.get("value")) or value != schema.get(
                "value"
            ):
                raise _fail(f"{field_name} differs from its frozen literal")
            return
        if schema_type == "nullable":
            if value is not None:
                _validate_schema(
                    value, schema.get("schema"), field_name=field_name
                )
            return
        # A literal mapping without a DSL tag is itself an exact object shape.
        if schema_type is None:
            mapping = _strict_mapping(value, set(schema), field_name=field_name)
            for name, child_schema in schema.items():
                _validate_schema(
                    mapping[name], child_schema, field_name=f"{field_name}.{name}"
                )
            return
    raise _fail(f"{field_name} has an unsupported frozen deep schema")


def _validate_registry_and_payload_schemas(
    bundle: VerifiedAuditByteBundle,
    *,
    allow_provisional_runtime: bool,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    if (
        _artifacts.artifact_schema_registry_sha256()
        != EXPECTED_ARTIFACT_SCHEMA_REGISTRY_SHA256
        or _artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
        != EXPECTED_ARTIFACT_SCHEMA_REGISTRY_SHA256
        or bundle.manifest["artifact_schema_registry_sha256"]
        != EXPECTED_ARTIFACT_SCHEMA_REGISTRY_SHA256
    ):
        raise _fail("artifact schema registry hash differs from the frozen module")
    registry = _artifacts.ARTIFACT_SCHEMA_REGISTRY
    descriptors = registry.get("json_artifacts")
    if not isinstance(descriptors, Mapping) or set(descriptors) != _NONCOPY_JSON_FILENAMES:
        raise _fail("artifact registry JSON inventory differs from the contract")
    parsed: dict[str, Mapping[str, Any]] = {}
    for filename in sorted(_NONCOPY_JSON_FILENAMES):
        try:
            value = _artifacts.parse_canonical_json_line(
                bundle.files[filename], field=filename
            )
        except _artifacts.ContextualExpertAggregation2024AuditArtifactError as exc:
            raise _fail(f"{filename} is not canonical strict JSON") from exc
        if not isinstance(value, Mapping):
            raise _fail(f"{filename} must be a JSON object")
        descriptor = descriptors[filename]
        if not isinstance(descriptor, Mapping):
            raise _fail(f"{filename} registry descriptor is malformed")
        _validate_schema(
            value,
            descriptor.get("literal_nested_schema"),
            field_name=filename,
        )
        parsed[filename] = dict(value)
    runtime = _runtime_payload(
        parsed["audit_runtime_cost_evidence.json"],
        allow_provisional_runtime=allow_provisional_runtime,
    )
    for filename in _artifacts.TABLE_SCHEMA_BY_FILENAME:
        try:
            _artifacts.parse_registered_table_bytes(
                filename, bundle.files[filename]
            )
        except Exception as exc:
            raise _fail(f"{filename} violates its frozen canonical table schema") from exc
    return parsed, runtime


def _validate_attempt_lock(value: Any) -> dict[str, Any]:
    lock = _strict_mapping(value, _LOCK_FIELDS, field_name="attempt lock")
    if (
        lock["lock_schema_version"] != _control.ATTEMPT_LOCK_SCHEMA_VERSION
        or lock["contract_version"] != _artifacts.CONTRACT_VERSION
        or lock["run_id"] != _artifacts.AUDIT_RUN_ID
        or lock["stage"] != _artifacts.AUDIT_STAGE
        or lock["attempt_number"] != 1
        or type(lock["attempt_number"]) is not int
        or lock["one_run_no_retry"] is not True
        or lock["persistent_after_success_or_failure"] is not True
        or lock["expected_branch"] != _artifacts.EXPECTED_BRANCH
        or lock["preregistration_commit"] != _artifacts.PREREGISTRATION_COMMIT
        or lock["artifact_schema_registry_sha256"]
        != _artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
        or lock["created_before_receipt_or_input_worktree_read"] is not True
        or lock["created_before_any_2024_market_value_release"] is not True
    ):
        raise _fail("attempt lock frozen identity or one-way claims changed")
    if type(lock["git_commit"]) is not str or _COMMIT_RE.fullmatch(
        lock["git_commit"]
    ) is None:
        raise _fail("attempt lock Git commit is malformed")
    if type(lock["dependency_identity_sha256"]) is not str or _SHA256_RE.fullmatch(
        lock["dependency_identity_sha256"]
    ) is None:
        raise _fail("attempt lock dependency identity hash is malformed")
    if not isinstance(lock["git_identity"], Mapping) or not isinstance(
        lock["parent_verification"], Mapping
    ):
        raise _fail("attempt lock Git or parent evidence is malformed")
    try:
        created = datetime.fromisoformat(lock["created_at_utc"])
    except (TypeError, ValueError) as exc:
        raise _fail("attempt lock creation timestamp is invalid") from exc
    if created.tzinfo is None:
        raise _fail("attempt lock creation timestamp lacks timezone")
    expected_paths = {
        "control_root": _artifacts.CONTROL_ROOT.as_posix(),
        "runs": _artifacts.OUTPUT_PARENT.as_posix(),
        "private": _artifacts.PRIVATE_DIRECTORY.as_posix(),
        "failure": _artifacts.FAILED_DIRECTORY.as_posix(),
        "final": _artifacts.OUTPUT_DIRECTORY.as_posix(),
        "pending_marker": _artifacts.PENDING_SUCCESS_MARKER_PATH.as_posix(),
        "success_marker": _artifacts.SUCCESS_MARKER_PATH.as_posix(),
    }
    if lock["output_paths"] != expected_paths:
        raise _fail("attempt lock output paths changed")
    return dict(lock)


def _persistent_copy_checks(
    root: Path, bundle: VerifiedAuditByteBundle
) -> dict[str, Any]:
    try:
        lock_value = _artifacts.parse_canonical_json_line(
            bundle.files[_artifacts.ATTEMPT_LOCK_FILENAME], field="attempt lock"
        )
    except _artifacts.ContextualExpertAggregation2024AuditArtifactError as exc:
        raise _fail("sealed attempt lock is not canonical JSON") from exc
    lock = _validate_attempt_lock(lock_value)
    copies = (
        (
            root / _artifacts.ATTEMPT_LOCK_PATH,
            _artifacts.ATTEMPT_LOCK_FILENAME,
            "persistent attempt lock",
        ),
        (
            root / _input.RECEIPT_PATH,
            "input_snapshot_receipt.json",
            "persistent sanitized receipt",
        ),
        (
            root / _input.INPUT_PATH,
            "audit_prices_through_2024.csv",
            "persistent bounded input",
        ),
    )
    for path, filename, label in copies:
        if _read_ordinary(path, field_name=label) != bundle.files[filename]:
            raise _fail(f"{label} differs from its sealed byte-exact copy")
    return lock


def _live_git_checks(
    root: Path,
    *,
    lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        live = _control.attest_prelock_git(root)
    except Exception as exc:
        raise _fail("live pushed Git/dependency attestation failed") from exc
    if (
        dict(live) != dict(lock["git_identity"])
        or dict(live) != dict(manifest["git_identity"])
        or live.get("commit") != lock["git_commit"]
        or live.get("dependency_identity_sha256")
        != lock["dependency_identity_sha256"]
    ):
        raise _fail("live Git/dependency identity differs from lock and manifest")
    expected_inventory = {
        "control_root": ["authorized_inputs", _input.RECEIPT_PATH.name],
        "authorized_inputs": [_input.INPUT_PATH.name],
        "runs": [],
    }
    if lock["prelock_control_inventory"] != expected_inventory:
        raise _fail("attempt lock prelock control inventory changed")
    expected_input = {
        **dict(live["input_git_identity"]),
        "expected_raw_sha256": _input.INPUT_RAW_SHA256,
        "expected_canonical_sha256": _input.INPUT_CANONICAL_SHA256,
        "expected_prefix_canonical_sha256": _input.PREFIX_CANONICAL_SHA256,
    }
    expected_receipt = {
        **dict(live["receipt_git_identity"]),
        "expected_raw_sha256": _input.RECEIPT_RAW_SHA256,
        "sealed_preparation_lineage": dict(
            _input.RECEIPT_EXPECTED["sealed_preparation_lineage"]
        ),
    }
    if (
        lock["input_identity"] != expected_input
        or lock["receipt_identity"] != expected_receipt
    ):
        raise _fail("attempt lock input or receipt Git identity changed")
    return live


def _success_marker(
    root: Path,
    bundle: VerifiedAuditByteBundle,
    *,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    marker_path = root / _artifacts.SUCCESS_MARKER_PATH
    try:
        marker = _artifacts.parse_success_marker_bytes(
            _read_ordinary(marker_path, field_name="durable success marker")
        )
    except _artifacts.ContextualExpertAggregation2024AuditArtifactError as exc:
        raise _fail("durable success marker is invalid") from exc
    manifest = bundle.manifest
    if (
        marker["attempt_lock_file_sha256"]
        != _sha256_bytes(bundle.files[_artifacts.ATTEMPT_LOCK_FILENAME])
        or marker["stage_manifest_file_sha256"]
        != _sha256_bytes(bundle.manifest_bytes)
        or marker["stage_manifest_self_sha256"] != manifest["manifest_sha256"]
        or marker["checksums_file_sha256"]
        != _sha256_bytes(bundle.checksums_bytes)
        or marker["git_commit"] != manifest["git_identity"].get("commit")
        or marker["marker_preparation_elapsed_seconds"]
        <= runtime["samples"]["prepromotion"]
    ):
        raise _fail("success marker does not bind the exact final byte bundle")
    return marker


def _integrity_checks(value: Mapping[str, Any]) -> Mapping[str, bool]:
    checks = value.get("checks")
    if not isinstance(checks, Mapping) or not checks:
        raise _fail("integrity evidence lacks its exact nonempty check mapping")
    result: dict[str, bool] = {}
    for name, passed in checks.items():
        if type(name) is not str or not name:
            raise _fail("integrity check name is invalid")
        result[name] = _exact_bool(passed, field_name=f"integrity.{name}")
    return result


def _semantic_section(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    child = value.get(name)
    return child if isinstance(child, Mapping) else value


def _direct_semantic_reconstruction(
    root: Path,
    bundle: VerifiedAuditByteBundle,
    parsed: Mapping[str, Mapping[str, Any]],
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently rebuild parent, input, replay, ledgers, XORs and gates."""

    try:
        parent = _parent.verify_parent_bundle(root)
    except Exception as exc:
        raise _fail("independent parent bundle verification failed") from exc
    if parsed["parent_verification.json"] != parent.lock_payload():
        raise _fail("sealed parent verification differs from independent regeneration")
    if lock["parent_verification"] != parent.lock_payload():
        raise _fail("attempt lock parent identity differs from regeneration")
    if (
        bundle.files["parent_stage_manifest.json"] != parent.manifest_bytes
        or bundle.files["parent_checksums.json"] != parent.checksums_bytes
    ):
        raise _fail("sealed parent manifest/checksums copies changed")
    try:
        bounded = _input.load_postlock_bounded_snapshot(
            input_path=bundle.directory / "audit_prices_through_2024.csv",
            receipt_path=bundle.directory / "input_snapshot_receipt.json",
            checkpoint=parent.checkpoint,
        )
    except Exception as exc:
        raise _fail("independent bounded-input verification failed") from exc
    observed_prefix = parsed["prefix_continuity_proof.json"]
    for name, expected in bounded.prefix_continuity.items():
        if name not in observed_prefix or observed_prefix[name] != expected:
            raise _fail("prefix continuity proof differs from regeneration")
    try:
        computation = _computation.compute_2024_continuation(
            parent=parent, bounded=bounded
        )
    except Exception as exc:
        raise _fail("independent chronological 2024 computation failed") from exc
    table_frames: dict[str, pd.DataFrame] = {
        "audit_fixed_features_2024.table.json": computation.fixed_features,
        "audit_forecast__fixed_comparators.table.json": (
            computation.fixed_comparator_forecast
        ),
        "audit_state_weight_diagnostics_2024.table.json": (
            computation.state_weight_diagnostics
        ),
    }
    for scenario in _artifacts.SCENARIO_ORDER:
        table_frames[f"audit_forecast__{scenario}.table.json"] = (
            computation.forecasts[scenario]
        )
        table_frames[f"audit_matured_lessons__{scenario}.table.json"] = (
            computation.matured_lessons[scenario]
        )
    for cost in _artifacts.COST_ORDER:
        for policy in _artifacts.POLICY_ORDER:
            table_frames[f"audit_ledger__{cost}__{policy}.table.json"] = (
                computation.ledgers[cost][policy]
            )
    for filename, frame in table_frames.items():
        try:
            expected = _artifacts.canonical_registered_table_bytes(filename, frame)
        except Exception as exc:
            raise _fail(f"independent table reconstruction failed: {filename}") from exc
        if expected != bundle.files[filename]:
            raise _fail(f"sealed table differs from independent replay: {filename}")
    checks = _integrity_checks(parsed["audit_integrity_evidence.json"])
    try:
        from . import contextual_expert_aggregation_2024_audit_runner as runner

        lock_evidence = _control.AttemptLockEvidence(
            path=bundle.directory / _artifacts.ATTEMPT_LOCK_FILENAME,
            content=dict(lock),
            bytes_value=bundle.files[_artifacts.ATTEMPT_LOCK_FILENAME],
            raw_sha256=_sha256_bytes(
                bundle.files[_artifacts.ATTEMPT_LOCK_FILENAME]
            ),
        )
        recomputed_checks = runner._integrity_checks(
            computation=computation,
            parent=parent,
            bounded=bounded,
            lock=lock_evidence,
            git_identity=lock["git_identity"],
        )
    except Exception as exc:
        raise _fail("independent fatal-integrity reconstruction failed") from exc
    if checks != recomputed_checks:
        raise _fail("sealed fatal-integrity checks differ from regeneration")
    for name, expected in computation.integrity_proofs.items():
        if name not in checks or checks[name] is not bool(expected):
            raise _fail("integrity evidence differs from computation proofs")
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
    try:
        evaluated = _evaluation.evaluate_2024_audit(
            policy_evidence,
            xor_evidence=computation.xors,
            integrity_checks=recomputed_checks,
        )
    except Exception as exc:
        raise _fail("independent unrounded evaluation failed") from exc
    metrics = _semantic_section(parsed["audit_metrics.json"], "metrics")
    gates = _semantic_section(parsed["audit_gate_report.json"], "gate_report")
    if metrics != evaluated["metrics"] or gates != evaluated["gate_report"]:
        raise _fail("sealed metrics or gates differ from independent evaluation")
    return {
        "parent": parent,
        "bounded": bounded,
        "computation": computation,
        "evaluation": evaluated,
        "integrity_checks": dict(recomputed_checks),
    }


def _regenerate_payloads(
    repo_root: Path,
    directory: Path,
    runtime_payload: Mapping[str, Any],
    *,
    semantic: Mapping[str, Any] | None = None,
) -> Mapping[str, bytes]:
    """Late-bound runner serialization seam; tests may monkeypatch this only."""

    try:
        from . import contextual_expert_aggregation_2024_audit_runner as runner
    except ImportError as exc:
        raise _fail("audit runner payload regenerator is unavailable") from exc
    lock_bytes = _read_ordinary(
        directory / _artifacts.ATTEMPT_LOCK_FILENAME,
        field_name="sealed attempt lock",
    )
    receipt_bytes = _read_ordinary(
        directory / "input_snapshot_receipt.json",
        field_name="sealed sanitized receipt",
    )
    input_bytes = _read_ordinary(
        directory / "audit_prices_through_2024.csv",
        field_name="sealed bounded input",
    )
    try:
        if semantic is not None:
            required = {
                "parent",
                "bounded",
                "computation",
                "evaluation",
                "integrity_checks",
            }
            if not isinstance(semantic, Mapping) or set(semantic) != required:
                raise _fail("semantic regeneration context has the wrong inventory")
            lock_value = _artifacts.parse_canonical_json_line(
                lock_bytes, field="sealed attempt lock"
            )
            if not isinstance(lock_value, Mapping) or not isinstance(
                lock_value.get("git_identity"), Mapping
            ):
                raise _fail("sealed attempt lock lacks its Git identity")
            lock_evidence = _control.AttemptLockEvidence(
                path=directory / _artifacts.ATTEMPT_LOCK_FILENAME,
                content=dict(lock_value),
                bytes_value=lock_bytes,
                raw_sha256=_sha256_bytes(lock_bytes),
            )
            build = getattr(runner, "_build_reproducible_payloads", None)
            if not callable(build):
                raise _fail("audit runner lacks the frozen payload builder")
            return build(
                parent=semantic["parent"],
                bounded=semantic["bounded"],
                lock=lock_evidence,
                git_identity=lock_value["git_identity"],
                computation=semantic["computation"],
                checks=semantic["integrity_checks"],
                evaluation=semantic["evaluation"],
                runtime_payload=dict(runtime_payload),
            )
        regenerate = getattr(runner, "reconstruct_reproducible_payloads", None)
        if not callable(regenerate):
            raise _fail("audit runner lacks the frozen payload regeneration API")
        return regenerate(
            Path(repo_root),
            Path(directory),
            lock_bytes=lock_bytes,
            receipt_bytes=receipt_bytes,
            input_bytes=input_bytes,
            runtime_payload=dict(runtime_payload),
        )
    except Exception as exc:
        raise _fail("independent payload regeneration failed") from exc


def _compare_regenerated_payloads(
    bundle: VerifiedAuditByteBundle,
    regenerated: Any,
) -> None:
    if not isinstance(regenerated, Mapping) or set(regenerated) != set(
        _artifacts.PAYLOAD_FILE_ORDER
    ):
        raise _fail("payload regeneration produced the wrong exact 41-file inventory")
    for filename in _artifacts.PAYLOAD_FILE_ORDER:
        expected = regenerated[filename]
        if type(expected) is not bytes:
            raise _fail(f"regenerated payload is not exact bytes: {filename}")
        if expected != bundle.files[filename]:
            raise _fail(f"sealed payload differs from regeneration: {filename}")


def _semantic_consistency(
    bundle: VerifiedAuditByteBundle,
    parsed: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    gate = _semantic_section(parsed["audit_gate_report.json"], "gate_report")
    report = parsed["report.json"]
    status = gate.get("status")
    stage_pass = gate.get("stage_pass")
    learning = gate.get("learning_classification")
    if (
        type(status) is not str
        or type(stage_pass) is not bool
        or learning not in _evaluation.LEARNING_STATUSES
        or status != bundle.manifest["status"]
        or stage_pass is not bundle.manifest["stage_pass"]
        or learning != bundle.manifest["learning_classification"]
    ):
        raise _fail("manifest status, gate report, and learning classification disagree")
    if "status" in report and report["status"] != status:
        raise _fail("human report status differs from the gate report")
    if "stage_pass" in report and report["stage_pass"] is not stage_pass:
        raise _fail("human report stage pass differs from the gate report")
    return status, learning


def _manifest_semantics(
    manifest: Mapping[str, Any], *, live_git: Mapping[str, Any]
) -> None:
    if (
        manifest["evidence_classification"]
        != _evaluation.EVIDENCE_CLASSIFICATION
        or manifest["receipt_file_sha256"] != _input.RECEIPT_RAW_SHA256
        or manifest["input_raw_sha256"] != _input.INPUT_RAW_SHA256
        or manifest["input_canonical_sha256"] != _input.INPUT_CANONICAL_SHA256
        or manifest["git_identity"] != dict(live_git)
        or manifest["dependency_identity_sha256"]
        != live_git["dependency_identity_sha256"]
    ):
        raise _fail("manifest semantic identities differ from live reconstruction")


def _verification_evidence(
    *,
    bundle: VerifiedAuditByteBundle,
    mode: str,
    marker: Mapping[str, Any] | None,
    status: str,
    learning: str,
    clock_samples: Mapping[str, float] | None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "verifier_id": VERIFIER_ID,
        "verified": True,
        "verification_mode": mode,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "stage": _artifacts.AUDIT_STAGE,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage_pass": bool(bundle.manifest["stage_pass"]),
        "status": status,
        "learning_classification": learning,
        "manifest_file_sha256": _sha256_bytes(bundle.manifest_bytes),
        "manifest_self_sha256": bundle.manifest["manifest_sha256"],
        "checksums_file_sha256": _sha256_bytes(bundle.checksums_bytes),
        "attempt_lock_file_sha256": _sha256_bytes(
            bundle.files[_artifacts.ATTEMPT_LOCK_FILENAME]
        ),
        "artifact_schema_registry_sha256": (
            _artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
        ),
        "exact_bundle_file_count": len(bundle.files),
        "later_market_data_accessed": False,
        "prior_2024_artifact_accessed": False,
        "network_access": False,
        "news_access": False,
        "llm_calls": 0,
        "api_calls": 0,
        "external_model_calls": 0,
        "external_cost_usd": 0.0,
        "verifier_clock_samples": dict(clock_samples or {}),
        "success_marker_sha256": (
            marker["marker_sha256"] if marker is not None else None
        ),
    }
    evidence["semantic_evidence_sha256"] = _sha256_json(evidence)
    return evidence


def _verify_bundle(
    directory: Path,
    *,
    repo_root: Path,
    mode: str,
    expected_manifest_sha256: str | None,
    clock: Any,
    allow_provisional_runtime: bool,
) -> dict[str, Any]:
    entry_phase = "private_preverify" if mode == PRIVATE_MODE else "preverify"
    post_phase = (
        "private_postreconstruction"
        if mode == PRIVATE_MODE
        else "postreconstruction"
    )
    exit_phase = "private_exit"
    _clock_check(clock, entry_phase)
    root, bundle_path = _bundle_location(
        Path(directory), repo_root=Path(repo_root), mode=mode
    )
    if mode == STANDALONE_MODE and expected_manifest_sha256 is None:
        try:
            early_marker = _artifacts.parse_success_marker_bytes(
                _read_ordinary(
                    root / _artifacts.SUCCESS_MARKER_PATH,
                    field_name="durable success marker",
                )
            )
        except _artifacts.ContextualExpertAggregation2024AuditArtifactError as exc:
            raise _fail("durable success marker is invalid") from exc
        expected_manifest_sha256 = early_marker["stage_manifest_self_sha256"]
    bundle = _read_bundle_bytes(
        bundle_path, expected_manifest_sha256=expected_manifest_sha256
    )
    _verify_control_layout(root, mode=mode, bundle=bundle_path)
    if (
        bundle.manifest["later_market_data_accessed"] is not False
        or bundle.manifest["prior_2024_artifact_accessed"] is not False
        or bundle.manifest["external_cost_usd"] != 0.0
    ):
        raise _fail("manifest records forbidden later/prior-result access or cost")
    lock = _persistent_copy_checks(root, bundle)
    if (
        lock["git_commit"] != bundle.manifest["git_identity"].get("commit")
        or lock["git_identity"] != bundle.manifest["git_identity"]
        or lock["dependency_identity_sha256"]
        != bundle.manifest["dependency_identity_sha256"]
        or lock["artifact_schema_registry_sha256"]
        != bundle.manifest["artifact_schema_registry_sha256"]
    ):
        raise _fail("attempt lock does not bind manifest Git/dependency identity")
    live_git = _live_git_checks(root, lock=lock, manifest=bundle.manifest)
    _manifest_semantics(bundle.manifest, live_git=live_git)
    parsed, runtime = _validate_registry_and_payload_schemas(
        bundle, allow_provisional_runtime=allow_provisional_runtime
    )
    # These direct calls make evaluation independence explicit.  The separate
    # payload regenerator then freezes all runner-owned envelopes/checkpoints.
    semantic = _direct_semantic_reconstruction(root, bundle, parsed, lock=lock)
    regenerated = _regenerate_payloads(
        root, bundle_path, runtime, semantic=semantic
    )
    _compare_regenerated_payloads(bundle, regenerated)
    status, learning = _semantic_consistency(bundle, parsed)
    _clock_check(clock, post_phase)
    marker = (
        _success_marker(root, bundle, runtime=runtime)
        if mode == STANDALONE_MODE
        else None
    )
    if mode == PRIVATE_MODE:
        _clock_check(clock, exit_phase)
    samples = clock.samples if isinstance(clock, VerificationClock) else None
    return _verification_evidence(
        bundle=bundle,
        mode=mode,
        marker=marker,
        status=status,
        learning=learning,
        clock_samples=samples,
    )


def verify_private_audit_bundle(
    directory: Path,
    *,
    repo_root: Path,
    expected_manifest_sha256: str,
    clock: Any | None = None,
    allow_provisional_runtime: bool = False,
) -> dict[str, Any]:
    """Read-only verification of the sole frozen private 43-file bundle."""

    if type(expected_manifest_sha256) is not str or _SHA256_RE.fullmatch(
        expected_manifest_sha256
    ) is None:
        raise _fail("private verification requires an exact manifest self-hash")
    selected_clock = VerificationClock.start() if clock is None else clock
    return _verify_bundle(
        Path(directory),
        repo_root=Path(repo_root),
        mode=PRIVATE_MODE,
        expected_manifest_sha256=expected_manifest_sha256,
        clock=selected_clock,
        allow_provisional_runtime=allow_provisional_runtime,
    )


def verify_audit(
    repo_root: Path | None = None,
    *,
    clock: Any | None = None,
) -> dict[str, Any]:
    """Read-only standalone verification of final bundle plus success marker."""

    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else _lexical(Path(repo_root))
    )
    selected_clock = VerificationClock.start() if clock is None else clock
    return _verify_bundle(
        root / _artifacts.OUTPUT_DIRECTORY,
        repo_root=root,
        mode=STANDALONE_MODE,
        expected_manifest_sha256=None,
        clock=selected_clock,
        allow_provisional_runtime=False,
    )


def verify_stage(stage: str) -> dict[str, Any]:
    if stage != _artifacts.AUDIT_STAGE:
        raise _fail("the verifier supports only the frozen audit_2024 stage")
    try:
        from .contextual_expert_aggregation_2024_audit_bootstrap import (
            require_bootstrap_deadline,
            require_active_attestation,
        )
    except ImportError as exc:
        raise _fail("audit bootstrap attestation is unavailable") from exc
    root = Path(__file__).resolve().parents[1]
    attestation = require_active_attestation(
        operation="verify", stage=stage, repo_root=root
    )

    class BootstrapClock:
        def check(self, phase: str) -> float:
            return require_bootstrap_deadline(attestation, phase)

    return verify_audit(root, clock=BootstrapClock())


__all__ = [
    "ContextualExpertAggregation2024AuditVerificationError",
    "EXPECTED_ARTIFACT_SCHEMA_REGISTRY_SHA256",
    "PRIVATE_MODE",
    "STANDALONE_MODE",
    "VERIFIER_ID",
    "VERIFIER_IMPLEMENTATION_PATH",
    "VERIFIER_LIMIT_SECONDS",
    "VerificationClock",
    "verify_audit",
    "verify_private_audit_bundle",
    "verify_stage",
]
