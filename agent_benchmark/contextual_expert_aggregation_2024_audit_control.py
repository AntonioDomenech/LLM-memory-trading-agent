"""One-way Git, path, dependency, and attempt-lock controls for the 2024 audit."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import contextual_expert_aggregation_2024_audit_artifacts as _artifacts
from . import contextual_expert_aggregation_2024_audit_input as _input
from . import contextual_expert_aggregation_experiment as _experiment


ATTEMPT_LOCK_SCHEMA_VERSION = "aapl-contextual-aggregation-2024-audit-lock-v2"
DEPENDENCY_IDENTITY_SCHEMA_VERSION = 1
_OBJECT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")

# The executable/control inventory is intentionally explicit and excludes the
# input, receipt, documentary snapshot note, quarantine, comparison table, and
# every result directory.  The tuple is expanded to a concrete sorted identity
# before it is hashed and copied into the attempt lock.
FROZEN_DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            *_experiment.FROZEN_DEPENDENCY_PATHS,
            Path("docs/aapl_causal_contextual_expert_aggregation_audit_v2.md"),
            Path("agent_benchmark/contextual_expert_aggregation_audit_artifacts.py"),
            Path("docs/aapl_causal_contextual_expert_aggregation_2024_audit_v2.md"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_artifacts.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_input.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_parent.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_replay.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_evidence.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_evaluation.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_computation.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_control.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_runner.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_verifier.py"),
            Path("agent_benchmark/contextual_expert_aggregation_2024_audit_bootstrap.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_artifacts.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_input.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_parent.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_replay.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_evidence.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_evaluation.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_computation.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_control.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_runner.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_verifier.py"),
            Path("tests/test_contextual_expert_aggregation_2024_audit_bootstrap.py"),
        )
    )
)


class AuditControlError(RuntimeError):
    """Raised when the one-way audit control surface changes."""


@dataclass(frozen=True)
class AttemptLockEvidence:
    path: Path
    content: Mapping[str, Any]
    bytes_value: bytes
    raw_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _canonical_bytes(value: Any) -> bytes:
    return _artifacts.canonical_json_line_bytes(value)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=check,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuditControlError("exact-path Git attestation failed") from exc


def _git_text(root: Path, *args: str) -> str:
    try:
        return _git(root, *args).stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise AuditControlError("Git attestation output is not strict UTF-8") from exc


def _literal(relative: str) -> str:
    if not relative or ".." in Path(relative).parts or any(c in relative for c in "\r\n\t"):
        raise AuditControlError("unsafe literal Git path")
    return f":(top,literal){relative}"


def _index_blob(root: Path, relative: str) -> str:
    output = _git_text(root, "ls-files", "--stage", "--", _literal(relative))
    match = re.fullmatch(r"[0-7]{6} ([0-9a-f]{40}|[0-9a-f]{64}) 0\t(.+)", output)
    if match is None or match.group(2) != relative:
        raise AuditControlError("tracked path lacks one ordinary stage-zero index entry")
    return match.group(1)


def _ordinary(path: Path, *, directory: bool, field: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise AuditControlError(f"{field} is missing or unreadable") from exc
    mode_ok = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    reparse = bool(getattr(info, "st_file_attributes", 0) & 0x0400)
    if not mode_ok or stat.S_ISLNK(info.st_mode) or reparse or path.is_mount():
        raise AuditControlError(f"{field} is not an ordinary non-reparse path")


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _descendant(path: Path, parent: Path, *, field: str) -> None:
    try:
        _lexical(path).relative_to(_lexical(parent))
    except ValueError as exc:
        raise AuditControlError(f"{field} escapes its frozen parent") from exc


def harden_runtime_paths(repo_root: Path) -> dict[str, str]:
    root = _lexical(repo_root)
    if ".." in repo_root.parts:
        raise AuditControlError("repository path contains a lexical parent traversal")
    _ordinary(root, directory=True, field="repository root")
    control = root / _artifacts.CONTROL_ROOT
    authorized = control / "authorized_inputs"
    input_path = root / _input.INPUT_PATH
    receipt_path = root / _input.RECEIPT_PATH
    for child, parent, name in (
        (control, root, "control root"),
        (authorized, control, "authorized input directory"),
        (input_path, authorized, "bounded input"),
        (receipt_path, control, "sanitized receipt"),
        (root / _artifacts.ATTEMPT_LOCK_PATH, control, "attempt lock"),
        (root / _artifacts.SUCCESS_MARKER_PATH, control, "success marker"),
        (root / _artifacts.PENDING_SUCCESS_MARKER_PATH, control, "pending marker"),
        (root / _artifacts.OUTPUT_PARENT, control, "runs directory"),
        (root / _artifacts.PRIVATE_DIRECTORY, root / _artifacts.OUTPUT_PARENT, "private directory"),
        (root / _artifacts.FAILED_DIRECTORY, root / _artifacts.OUTPUT_PARENT, "failure directory"),
        (root / _artifacts.OUTPUT_DIRECTORY, root / _artifacts.OUTPUT_PARENT, "final directory"),
    ):
        _descendant(child, parent, field=name)
    _ordinary(control, directory=True, field="control root")
    _ordinary(authorized, directory=True, field="authorized input directory")
    _ordinary(input_path, directory=False, field="bounded input")
    _ordinary(receipt_path, directory=False, field="sanitized receipt")
    for optional in (
        root / _artifacts.ATTEMPT_LOCK_PATH,
        root / _artifacts.SUCCESS_MARKER_PATH,
        root / _artifacts.PENDING_SUCCESS_MARKER_PATH,
        root / _artifacts.OUTPUT_PARENT,
        root / _artifacts.PRIVATE_DIRECTORY,
        root / _artifacts.FAILED_DIRECTORY,
        root / _artifacts.OUTPUT_DIRECTORY,
    ):
        if optional.exists():
            _ordinary(optional, directory=optional.suffix == "", field="existing audit output")
        else:
            parent = optional.parent
            while not parent.exists():
                parent = parent.parent
            _ordinary(parent, directory=True, field="nonexistent output ancestor")
    return {
        "repo_root": str(root),
        "control_root": str(control),
        "authorized_inputs": str(authorized),
        "input": str(input_path),
        "receipt": str(receipt_path),
        "attempt_lock": str(root / _artifacts.ATTEMPT_LOCK_PATH),
        "runs": str(root / _artifacts.OUTPUT_PARENT),
        "private": str(root / _artifacts.PRIVATE_DIRECTORY),
        "failure": str(root / _artifacts.FAILED_DIRECTORY),
        "final": str(root / _artifacts.OUTPUT_DIRECTORY),
        "pending_marker": str(root / _artifacts.PENDING_SUCCESS_MARKER_PATH),
        "success_marker": str(root / _artifacts.SUCCESS_MARKER_PATH),
    }


def control_inventory(repo_root: Path) -> dict[str, list[str]]:
    root = _lexical(repo_root)
    control = root / _artifacts.CONTROL_ROOT
    authorized = control / "authorized_inputs"
    try:
        root_names = sorted(entry.name for entry in os.scandir(control))
        input_names = sorted(entry.name for entry in os.scandir(authorized))
        run_names = (
            sorted(entry.name for entry in os.scandir(root / _artifacts.OUTPUT_PARENT))
            if (root / _artifacts.OUTPUT_PARENT).exists()
            else []
        )
    except OSError as exc:
        raise AuditControlError("bounded control inventory is unreadable") from exc
    return {
        "control_root": root_names,
        "authorized_inputs": input_names,
        "runs": run_names,
    }


def require_pristine_control_root(repo_root: Path) -> dict[str, list[str]]:
    inventory = control_inventory(repo_root)
    if inventory != {
        "control_root": ["authorized_inputs", "input_snapshot_receipt.json"],
        "authorized_inputs": ["aapl_spy_qqq_through_2024.csv"],
        "runs": [],
    }:
        raise AuditControlError("the sole 2024 audit attempt is already consumed or stale")
    return inventory


def _dependency_identity(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, str]] = {}
    for path in FROZEN_DEPENDENCY_PATHS:
        relative = path.as_posix()
        local_path = root / path
        _ordinary(local_path, directory=False, field="frozen dependency")
        head_blob = _git_text(root, "rev-parse", f"HEAD:{relative}")
        index_blob = _index_blob(root, relative)
        if head_blob != index_blob or _OBJECT_RE.fullmatch(head_blob) is None:
            raise AuditControlError("frozen dependency HEAD/index identity changed")
        if _git_text(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            _literal(relative),
        ):
            raise AuditControlError("frozen dependency has exact-path worktree changes")
        try:
            local = local_path.read_bytes()
            head = _git(root, "show", f"HEAD:{relative}").stdout
            indexed = _git(root, "show", f":{relative}").stdout
        except OSError as exc:
            raise AuditControlError("frozen dependency bytes are unreadable") from exc
        normalized = (
            local.replace(b"\r\n", b"\n")
            if path.suffix.lower() in {".py", ".pyw", ".md"}
            or relative in {".gitattributes", ".gitignore", "requirements.txt"}
            else local
        )
        if head != indexed or head != normalized:
            raise AuditControlError("frozen dependency bytes differ from HEAD/index")
        files[relative] = {
            "head_blob": head_blob,
            "index_blob": index_blob,
            "worktree_raw_sha256": _sha256_bytes(local),
        }
    identity = {
        "schema_version": DEPENDENCY_IDENTITY_SCHEMA_VERSION,
        "files": {key: files[key] for key in sorted(files)},
    }
    return identity


def _unopened_git_identity(root: Path, path: Path, *, expected_blob: str) -> dict[str, str]:
    relative = path.as_posix()
    head = _git_text(root, "rev-parse", f"HEAD:{relative}")
    indexed = _index_blob(root, relative)
    if head != expected_blob or indexed != expected_blob:
        raise AuditControlError("unopened audit control HEAD/index blob changed")
    return {"path": relative, "head_blob": head, "index_blob": indexed}


def attest_prelock_git(repo_root: Path) -> dict[str, Any]:
    root = _lexical(repo_root)
    actual = _lexical(Path(_git_text(root, "rev-parse", "--show-toplevel")))
    branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    commit = _git_text(root, "rev-parse", "HEAD")
    upstream = _git_text(
        root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    upstream_commit = _git_text(root, "rev-parse", "@{upstream}")
    origin = _git_text(root, "remote", "get-url", "origin")
    ancestry = _git(
        root,
        "merge-base",
        "--is-ancestor",
        _artifacts.PREREGISTRATION_COMMIT,
        commit,
        check=False,
    )
    if (
        actual != root
        or branch != _artifacts.EXPECTED_BRANCH
        or _OBJECT_RE.fullmatch(commit) is None
        or upstream != f"origin/{branch}"
        or upstream_commit != commit
        or origin not in _experiment.EXPECTED_ORIGIN_URLS
        or ancestry.returncode != 0
    ):
        raise AuditControlError("audit requires the exact pushed runtime branch and ancestry")
    dependencies = _dependency_identity(root)
    return {
        "branch": branch,
        "commit": commit,
        "upstream": upstream,
        "upstream_commit": upstream_commit,
        "origin_url": origin,
        "origin_repository": _experiment.EXPECTED_ORIGIN_REPOSITORY,
        "preregistration_commit": _artifacts.PREREGISTRATION_COMMIT,
        "preregistration_is_ancestor": True,
        "head_equals_upstream": True,
        "dependency_identity": dependencies,
        "dependency_identity_sha256": _sha256_json(dependencies),
        "input_git_identity": _unopened_git_identity(
            root, _input.INPUT_PATH, expected_blob=_input.INPUT_GIT_BLOB
        ),
        "receipt_git_identity": _unopened_git_identity(
            root, _input.RECEIPT_PATH, expected_blob=_input.RECEIPT_GIT_BLOB
        ),
        "prelock_input_worktree_bytes_opened": False,
        "prelock_receipt_worktree_bytes_opened": False,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def create_attempt_lock(
    *,
    repo_root: Path,
    git_identity: Mapping[str, Any],
    parent_verification: Mapping[str, Any],
    pristine_inventory: Mapping[str, Any],
) -> AttemptLockEvidence:
    root = _lexical(repo_root)
    lock_path = root / _artifacts.ATTEMPT_LOCK_PATH
    if lock_path.exists() or (root / _artifacts.OUTPUT_PARENT).exists():
        raise AuditControlError("the sole stage attempt has already been consumed")
    content = {
        "lock_schema_version": ATTEMPT_LOCK_SCHEMA_VERSION,
        "contract_version": _artifacts.CONTRACT_VERSION,
        "run_id": _artifacts.AUDIT_RUN_ID,
        "stage": _artifacts.AUDIT_STAGE,
        "attempt_number": 1,
        "one_run_no_retry": True,
        "persistent_after_success_or_failure": True,
        "expected_branch": _artifacts.EXPECTED_BRANCH,
        "preregistration_commit": _artifacts.PREREGISTRATION_COMMIT,
        "git_commit": git_identity["commit"],
        "git_identity": dict(git_identity),
        "dependency_identity_sha256": git_identity["dependency_identity_sha256"],
        "artifact_schema_registry_sha256": _artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256,
        "parent_verification": dict(parent_verification),
        "input_identity": {
            **dict(git_identity["input_git_identity"]),
            "expected_raw_sha256": _input.INPUT_RAW_SHA256,
            "expected_canonical_sha256": _input.INPUT_CANONICAL_SHA256,
            "expected_prefix_canonical_sha256": _input.PREFIX_CANONICAL_SHA256,
        },
        "receipt_identity": {
            **dict(git_identity["receipt_git_identity"]),
            "expected_raw_sha256": _input.RECEIPT_RAW_SHA256,
            "sealed_preparation_lineage": dict(
                _input.RECEIPT_EXPECTED["sealed_preparation_lineage"]
            ),
        },
        "prelock_control_inventory": dict(pristine_inventory),
        "output_paths": {
            "control_root": _artifacts.CONTROL_ROOT.as_posix(),
            "runs": _artifacts.OUTPUT_PARENT.as_posix(),
            "private": _artifacts.PRIVATE_DIRECTORY.as_posix(),
            "failure": _artifacts.FAILED_DIRECTORY.as_posix(),
            "final": _artifacts.OUTPUT_DIRECTORY.as_posix(),
            "pending_marker": _artifacts.PENDING_SUCCESS_MARKER_PATH.as_posix(),
            "success_marker": _artifacts.SUCCESS_MARKER_PATH.as_posix(),
        },
        "created_before_receipt_or_input_worktree_read": True,
        "created_before_any_2024_market_value_release": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = _canonical_bytes(content)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _experiment._fsync_directory(lock_path.parent)
        persisted = lock_path.read_bytes()
    except OSError as exc:
        raise AuditControlError("durable attempt lock creation failed closed") from exc
    if persisted != payload:
        raise AuditControlError("durable attempt lock changed during readback")
    return AttemptLockEvidence(
        path=lock_path,
        content=content,
        bytes_value=payload,
        raw_sha256=_sha256_bytes(payload),
    )


def read_attempt_lock(repo_root: Path) -> AttemptLockEvidence:
    path = _lexical(repo_root) / _artifacts.ATTEMPT_LOCK_PATH
    _ordinary(path, directory=False, field="durable attempt lock")
    try:
        payload = path.read_bytes()
        value = _artifacts.parse_canonical_json_line(payload)
    except (OSError, _artifacts.ContextualExpertAggregation2024AuditArtifactError) as exc:
        raise AuditControlError("durable attempt lock is unreadable or noncanonical") from exc
    if not isinstance(value, dict):
        raise AuditControlError("durable attempt lock is not a JSON object")
    return AttemptLockEvidence(
        path=path,
        content=value,
        bytes_value=payload,
        raw_sha256=_sha256_bytes(payload),
    )


def reattest_after_lock(
    repo_root: Path,
    *,
    expected_git_identity: Mapping[str, Any],
    expected_lock: AttemptLockEvidence,
) -> None:
    persisted = read_attempt_lock(repo_root)
    if (
        persisted.bytes_value != expected_lock.bytes_value
        or persisted.raw_sha256 != expected_lock.raw_sha256
        or persisted.content != dict(expected_lock.content)
    ):
        raise AuditControlError("attempt lock changed after durable creation")
    current = attest_prelock_git(repo_root)
    if current != dict(expected_git_identity):
        raise AuditControlError("Git or dependency identity changed after the lock")
    inventory = control_inventory(repo_root)
    if inventory != {
        "control_root": [
            _artifacts.ATTEMPT_LOCK_FILENAME,
            "authorized_inputs",
            "input_snapshot_receipt.json",
        ],
        "authorized_inputs": ["aapl_spy_qqq_through_2024.csv"],
        "runs": [],
    }:
        raise AuditControlError("control-root inventory changed after the lock")


def read_postlock_tracked_file(
    repo_root: Path,
    *,
    relative_path: Path,
    expected_blob: str,
    expected_raw_sha256: str,
) -> bytes:
    root = _lexical(repo_root)
    path = root / relative_path
    _ordinary(path, directory=False, field="post-lock tracked control")
    relative = relative_path.as_posix()
    if (
        _git_text(root, "rev-parse", f"HEAD:{relative}") != expected_blob
        or _index_blob(root, relative) != expected_blob
        or _git_text(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            _literal(relative),
        )
    ):
        raise AuditControlError("post-lock tracked control identity or status changed")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AuditControlError("post-lock tracked control became unreadable") from exc
    if _sha256_bytes(payload) != expected_raw_sha256:
        raise AuditControlError("post-lock tracked control raw hash changed")
    return payload


__all__ = [
    "ATTEMPT_LOCK_SCHEMA_VERSION",
    "FROZEN_DEPENDENCY_PATHS",
    "AuditControlError",
    "AttemptLockEvidence",
    "harden_runtime_paths",
    "control_inventory",
    "require_pristine_control_root",
    "attest_prelock_git",
    "create_attempt_lock",
    "read_attempt_lock",
    "reattest_after_lock",
    "read_postlock_tracked_file",
]
