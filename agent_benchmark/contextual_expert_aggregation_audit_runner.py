"""One-shot post-rejection 2019-2023 continuation audit runner.

The module deliberately separates the pre-lock authorization path from the
post-lock data path.  No function on the pre-lock path opens the through-2023
worktree file.  Model and ledger primitives are inherited unchanged from v2.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import contextual_expert_aggregation_artifacts as _v2_artifacts
from . import contextual_expert_aggregation_evaluation as _v2_evaluation
from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_ledger as _ledger
from . import contextual_expert_aggregation_replay as _replay
from . import contextual_expert_aggregation_stage as _v2_stage
from .contextual_expert_aggregation_audit_artifacts import (
    ATTEMPT_LOCK_FILENAME,
    ATTEMPT_LOCK_PATH,
    AUDIT_PAYLOAD_NAMES,
    AUDIT_RUN_ID,
    AUDIT_STAGE,
    CONTRACT_VERSION,
    EXPECTED_BRANCH,
    GIT_ATTRIBUTES_BYTES,
    LAST_OBSERVED_SESSION,
    OUTPUT_DIRECTORY,
    OUTPUT_PARENT,
    build_audit_checkpoint,
    audit_checkpoint_bytes,
    seal_audit_bundle,
)


ATTEMPT_LOCK_SCHEMA_VERSION = (
    "contextual-expert-aggregation-post-rejection-audit-lock-v2"
)
EVIDENCE_CLASSIFICATION = "post_rejection_reused_historical_continuation_audit"
PARENT_DIRECTORY = Path(
    "e/aapl_causal_contextual_expert_aggregation_v2/"
    "contextual-expert-aggregation-development-v2"
)
PARENT_MANIFEST_SELF_SHA256 = (
    "sha256:6cb657d1d9c323bd96a84b598b9b993e2444f756a65521aa1bf28207f7a63539"
)
PARENT_CHECKPOINT_PAYLOAD_SHA256 = (
    "sha256:abe07112ce2bb72292d65b357dc006ab3be268f8b0d02077563c59fa71a22538"
)
PARENT_CHECKPOINT_SELF_SHA256 = (
    "sha256:e195356c0540a4a05ea635927fa466c1c85948beb7de0191e80f5bc10d60fd03"
)
PARENT_SEMANTIC_EVIDENCE_SHA256 = (
    "sha256:964ad44206d3a1cb481a330317217a2d910101b34e35240aa1186c8a6069e944"
)
AUDIT_INPUT_SPEC = _experiment.CONFIRMATION_PRICE_SPEC
AUDIT_INPUT_PATH = AUDIT_INPUT_SPEC.relative_path
DEVELOPMENT_INPUT_PATH = _experiment.DEVELOPMENT_PRICE_SPEC.relative_path
RUN_TIME_LIMIT_SECONDS = 3_600.0
ACCOUNT_START = "2005-01-01"
AUDIT_SUFFIX_START = pd.Timestamp("2019-01-01")
AUDIT_SUFFIX_END = pd.Timestamp("2023-12-31")
AUDIT_INTEGRITY_CHECKS = frozenset(
    set(_v2_evaluation.COMMON_INTEGRITY_CHECKS)
    | {
        "development_checkpoint_continuity_exact",
        "online_frozen_forecast_prefix_exact",
        "frozen_post_cutoff_state_unchanged",
        "audit_attempt_lock_exact",
        "exact_rejected_parent_authorization",
        "post_2023_market_values_not_accessed",
    }
)


FROZEN_DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            # Preserve the complete inherited v2 executable, contract, and test
            # inventory, including its bootstrap and transitive verifier imports.
            *_experiment.FROZEN_DEPENDENCY_PATHS,
            Path("e/APPROACH_COMPARISON.md"),
            Path("e/aapl_causal_contextual_expert_aggregation_v2/REJECTED.md"),
            Path("docs/aapl_causal_contextual_expert_aggregation_audit_v2.md"),
            Path("agent_benchmark/contextual_expert_aggregation_audit_parent.py"),
            Path("agent_benchmark/contextual_expert_aggregation_audit_evaluation.py"),
            Path("agent_benchmark/contextual_expert_aggregation_audit_artifacts.py"),
            Path("agent_benchmark/contextual_expert_aggregation_audit_runner.py"),
            Path("agent_benchmark/contextual_expert_aggregation_audit_verifier.py"),
            Path("agent_benchmark/contextual_expert_aggregation_audit_bootstrap.py"),
            Path("tests/test_contextual_expert_aggregation_audit_parent.py"),
            Path("tests/test_contextual_expert_aggregation_audit_evaluation.py"),
            Path("tests/test_contextual_expert_aggregation_audit_artifacts.py"),
            Path("tests/test_contextual_expert_aggregation_audit_runner.py"),
            Path("tests/test_contextual_expert_aggregation_audit_verifier.py"),
            Path("tests/test_contextual_expert_aggregation_audit_bootstrap.py"),
        )
    )
)


class ContextualExpertAggregationAuditError(RuntimeError):
    """Raised when an audit contract, firewall, replay, or seal check fails."""


@dataclass(frozen=True)
class AuditLockEvidence:
    path: str
    sha256: str
    content: Mapping[str, Any]
    bytes_value: bytes
    parent_directory_fsync_supported: bool | None


@dataclass(frozen=True)
class PreparedDevelopment:
    parent_bundle: _experiment.VerifiedBundle
    parent_verification: Mapping[str, Any]
    snapshot: _experiment.LoadedPriceSnapshot
    online_replay: _replay.ReplayResult
    checkpoint: _v2_artifacts.CompositeStageCheckpoint
    parent_manifest_bytes: bytes
    parent_checksums_bytes: bytes
    parent_replay_artifacts: Mapping[str, Any]


@dataclass(frozen=True)
class AuditComputation:
    snapshot: _experiment.LoadedPriceSnapshot
    replays: Mapping[str, _replay.ReplayResult]
    full_forecasts: Mapping[str, pd.DataFrame]
    full_matured_lessons: Mapping[str, pd.DataFrame]
    full_fixed_features: pd.DataFrame
    ledgers: Mapping[str, Mapping[str, pd.DataFrame]]
    accounts: Mapping[str, Mapping[str, _ledger.AccountState]]
    episodes: Mapping[str, Mapping[str, _ledger.EpisodeExtraction]]
    xors: Mapping[str, Mapping[str, _ledger.XorExtraction]]
    parent_causal_prefix_proof: Mapping[str, Any]
    source_bundle_provenance: Mapping[str, Any]
    replay_diagnostics: Mapping[str, Any]
    prefix_continuity_proof: Mapping[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ContextualExpertAggregationAuditError(
            "audit evidence contains a nonfinite number"
        )
    return value


def _json_bytes(value: Any) -> bytes:
    return _experiment.pretty_json_bytes(_jsonable(value))


def _sha256_json(value: Any) -> str:
    return _experiment.sha256_bytes(
        _experiment.canonical_json_bytes(_jsonable(value))
    )


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30.0,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContextualExpertAggregationAuditError(
            "audit Git inspection failed"
        ) from exc


def _git_text(root: Path, *args: str) -> str:
    try:
        return _git_bytes(root, *args).decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ContextualExpertAggregationAuditError(
            "audit Git output is not strict UTF-8"
        ) from exc


def _safe_relative(root: Path, path: Path, *, field: str) -> str:
    try:
        value = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContextualExpertAggregationAuditError(
            f"{field} must remain inside the repository"
        ) from exc
    if not value or any(character in value for character in "\r\n\t"):
        raise ContextualExpertAggregationAuditError(f"{field} is unsafe")
    return value


def _tracked_index_object_id(root: Path, relative: str) -> str:
    literal = f":(top,literal){relative}"
    value = _git_text(root, "ls-files", "--stage", "--", literal)
    match = re.fullmatch(r"[0-7]{6} ([0-9a-f]{40}|[0-9a-f]{64}) 0\t(.+)", value)
    if match is None or match.group(2) != relative:
        raise ContextualExpertAggregationAuditError(
            "audit input must have one ordinary stage-zero index entry"
        )
    return match.group(1)


def _dependency_identity(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for relative_path in FROZEN_DEPENDENCY_PATHS:
        relative = relative_path.as_posix()
        try:
            _git_bytes(root, "ls-files", "--error-unmatch", "--", relative)
            committed = _git_bytes(root, "show", f"HEAD:{relative}")
            indexed = _git_bytes(root, "show", f":{relative}")
            local_path = _experiment._harden_path(
                root / relative_path,
                label=f"Frozen audit dependency {relative}",
                require_exists=True,
            )
            if not local_path.is_file():
                raise ContextualExpertAggregationAuditError(
                    f"frozen audit dependency is not a regular file: {relative}"
                )
            local = local_path.read_bytes()
            head_blob = _git_text(root, "rev-parse", f"HEAD:{relative}")
            index_blob = _tracked_index_object_id(root, relative)
        except OSError as exc:
            raise ContextualExpertAggregationAuditError(
                f"frozen audit dependency is unreadable: {relative}"
            ) from exc
        normalized_local = (
            local.replace(b"\r\n", b"\n")
            if relative_path.suffix.lower() in {".py", ".pyw", ".md"}
            or relative in {".gitattributes", ".gitignore", "requirements.txt"}
            else local
        )
        if (
            committed != indexed
            or committed != normalized_local
            or head_blob != index_blob
        ):
            raise ContextualExpertAggregationAuditError(
                f"frozen audit dependency differs from HEAD/index: {relative}"
            )
        result[relative] = {
            "sha256": _experiment.sha256_bytes(committed),
            "git_blob": head_blob,
        }
    return result


def _status_excluding_authorized_paths(root: Path, *, after_lock: bool) -> str:
    input_relative = AUDIT_INPUT_PATH.as_posix()
    args = [
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        f":(top,exclude,literal){input_relative}",
    ]
    if after_lock:
        args.append(f":(top,exclude,literal){OUTPUT_PARENT.as_posix()}")
    return _git_text(root, *args)


def require_prelock_git_identity(repo_root: Path) -> dict[str, Any]:
    """Bind clean pushed code and the input object without opening input bytes."""

    root = repo_root.resolve()
    input_path = root / AUDIT_INPUT_PATH
    input_relative = _safe_relative(root, input_path, field="audit input")
    try:
        actual = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
        branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
        commit = _git_text(root, "rev-parse", "HEAD")
        upstream = _git_text(
            root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        upstream_commit = _git_text(root, "rev-parse", "@{upstream}")
        origin_url = _git_text(root, "remote", "get-url", "origin")
        input_head_blob = _git_text(root, "rev-parse", f"HEAD:{input_relative}")
        input_index_blob = _tracked_index_object_id(root, input_relative)
    except ContextualExpertAggregationAuditError:
        raise
    if actual != root:
        raise ContextualExpertAggregationAuditError(
            "repo_root is not the actual Git top level"
        )
    if branch != EXPECTED_BRANCH:
        raise ContextualExpertAggregationAuditError(
            "audit is running on the wrong frozen branch"
        )
    if (
        not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
        or upstream != f"origin/{branch}"
        or upstream_commit != commit
        or origin_url not in _experiment.EXPECTED_ORIGIN_URLS
    ):
        raise ContextualExpertAggregationAuditError(
            "audit requires attached HEAD at its exact pushed origin branch"
        )
    if _status_excluding_authorized_paths(root, after_lock=False):
        raise ContextualExpertAggregationAuditError(
            "audit requires a clean tree outside the unopened input"
        )
    if (
        input_head_blob != AUDIT_INPUT_SPEC.git_blob
        or input_index_blob != AUDIT_INPUT_SPEC.git_blob
    ):
        raise ContextualExpertAggregationAuditError(
            "audit input HEAD/index blob differs before the attempt lock"
        )
    dependencies = _dependency_identity(root)
    return {
        "branch": branch,
        "commit": commit,
        "upstream": upstream,
        "upstream_commit": upstream_commit,
        "origin_url": origin_url,
        "origin_repository": _experiment.EXPECTED_ORIGIN_REPOSITORY,
        "head_equals_upstream": True,
        "dirty": False,
        "cleanliness_scope": "all_paths_except_unopened_input",
        "prelock_input_worktree_bytes_opened": False,
        "input_path": input_relative,
        "input_head_object_id": input_head_blob,
        "input_index_object_id": input_index_blob,
        "input_head_index_equal": True,
        "tracked_dependency_identity": dependencies,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def require_postlock_input_identity(
    repo_root: Path,
    prelock: Mapping[str, Any],
    *,
    allowed_output_directories: Sequence[str] = (),
) -> dict[str, Any]:
    """After lock creation, bind input HEAD, index, path status, and bytes."""

    root = repo_root.resolve()
    try:
        output_parent = _experiment._harden_path(
            root / OUTPUT_PARENT,
            label="Audit output parent",
            require_exists=True,
        )
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise ContextualExpertAggregationAuditError(
            "audit output parent is redirected or missing after the lock"
        ) from exc
    if not output_parent.is_dir():
        raise ContextualExpertAggregationAuditError(
            "audit output parent is not a regular directory"
        )
    try:
        output_entries = list(output_parent.iterdir())
    except OSError as exc:
        raise ContextualExpertAggregationAuditError(
            "audit output parent is unreadable after the lock"
        ) from exc
    allowed_directories = set(allowed_output_directories)
    if any(
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name == ATTEMPT_LOCK_FILENAME
        for name in allowed_directories
    ):
        raise ContextualExpertAggregationAuditError(
            "post-lock allowed output directory inventory is unsafe"
        )
    if {entry.name for entry in output_entries} != {
        ATTEMPT_LOCK_FILENAME,
        *allowed_directories,
    }:
        raise ContextualExpertAggregationAuditError(
            "unexpected output exists beside the durable audit lock"
        )
    for entry in output_entries:
        if _experiment._path_is_reparse_point(entry):
            raise ContextualExpertAggregationAuditError(
                "audit output parent contains a redirected entry"
            )
        if entry.name == ATTEMPT_LOCK_FILENAME:
            if not entry.is_file():
                raise ContextualExpertAggregationAuditError(
                    "durable audit lock is not a regular file"
                )
        elif not entry.is_dir():
            raise ContextualExpertAggregationAuditError(
                "allowed audit output is not a regular directory"
            )
    if _status_excluding_authorized_paths(root, after_lock=True):
        raise ContextualExpertAggregationAuditError(
            "audit code tree changed after attempt-lock creation"
        )
    current_dependencies = _dependency_identity(root)
    if current_dependencies != prelock.get("tracked_dependency_identity"):
        raise ContextualExpertAggregationAuditError(
            "frozen dependency identity changed after the attempt lock"
        )
    current_commit = _git_text(root, "rev-parse", "HEAD")
    current_branch = _git_text(
        root, "symbolic-ref", "--quiet", "--short", "HEAD"
    )
    current_upstream = _git_text(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    current_upstream_commit = _git_text(root, "rev-parse", "@{upstream}")
    current_origin = _git_text(root, "remote", "get-url", "origin")
    if (
        current_commit != prelock.get("commit")
        or current_branch != prelock.get("branch")
        or current_upstream != prelock.get("upstream")
        or current_upstream_commit != prelock.get("upstream_commit")
        or current_origin != prelock.get("origin_url")
        or current_branch != EXPECTED_BRANCH
        or current_upstream_commit != current_commit
    ):
        raise ContextualExpertAggregationAuditError(
            "Git branch, commit, upstream, or origin changed after the lock"
        )
    input_path = root / AUDIT_INPUT_PATH
    identity = _experiment.tracked_file_identity(
        root,
        input_path,
        expected_sha256=AUDIT_INPUT_SPEC.raw_sha256,
        expected_git_blob=AUDIT_INPUT_SPEC.git_blob,
        require_literal_local_bytes=True,
    )
    literal = f":(top,literal){identity.path}"
    if _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        literal,
    ):
        raise ContextualExpertAggregationAuditError(
            "audit input path is not clean after the lock"
        )
    if (
        identity.path != prelock.get("input_path")
        or identity.git_blob != prelock.get("input_head_object_id")
        or identity.git_blob != prelock.get("input_index_object_id")
    ):
        raise ContextualExpertAggregationAuditError(
            "audit input identity changed across the lock boundary"
        )
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "git_blob": identity.git_blob,
        "head_index_local_equal": True,
        "path_status_clean": True,
        "verified_after_attempt_lock": True,
    }


def require_unused_destination(repo_root: Path) -> Path:
    root = repo_root.resolve()
    parent = root / OUTPUT_PARENT
    final = root / OUTPUT_DIRECTORY
    lock = root / ATTEMPT_LOCK_PATH
    temporary = parent / ".audit-v2.sealing"
    if parent.exists() or final.exists() or lock.exists() or temporary.exists():
        raise ContextualExpertAggregationAuditError(
            "the sole audit attempt already exists or has stale output"
        )
    return parent


def create_attempt_lock(
    *,
    repo_root: Path,
    prelock_git_identity: Mapping[str, Any],
    parent_verification: Mapping[str, Any],
) -> AuditLockEvidence:
    """Durably and irreversibly consume the sole attempt before input access."""

    root = repo_root.resolve()
    parent = root / OUTPUT_PARENT
    lock_path = root / ATTEMPT_LOCK_PATH
    if parent.exists() or lock_path.exists():
        raise ContextualExpertAggregationAuditError(
            "the one audit attempt has already been consumed"
        )
    try:
        expected_parent = _experiment._harden_path(
            parent.parent,
            label="Audit output grandparent",
            require_exists=True,
        )
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise ContextualExpertAggregationAuditError(
            "audit output grandparent is redirected or missing"
        ) from exc
    if not expected_parent.is_dir():
        raise ContextualExpertAggregationAuditError(
            "audit output grandparent is missing"
        )
    parent.mkdir()
    _experiment._fsync_directory(parent.parent)
    parent_fsync_supported = _experiment._fsync_directory(parent)
    content = {
        "schema_version": ATTEMPT_LOCK_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "stage": AUDIT_STAGE,
        "run_id": AUDIT_RUN_ID,
        "attempt_number": 1,
        "one_run_no_retry": True,
        "persistent_after_success_or_failure": True,
        "created_before_through_2023_worktree_read": True,
        "created_before_any_2019_market_value": True,
        "expected_branch": EXPECTED_BRANCH,
        "git_commit": prelock_git_identity.get("commit"),
        "input_path": AUDIT_INPUT_PATH.as_posix(),
        "input_git_blob": AUDIT_INPUT_SPEC.git_blob,
        "input_raw_sha256": AUDIT_INPUT_SPEC.raw_sha256,
        "parent_manifest_sha256": PARENT_MANIFEST_SELF_SHA256,
        "parent_checkpoint_payload_sha256": PARENT_CHECKPOINT_PAYLOAD_SHA256,
        "parent_checkpoint_self_sha256": PARENT_CHECKPOINT_SELF_SHA256,
        "parent_semantic_evidence_sha256": PARENT_SEMANTIC_EVIDENCE_SHA256,
        "parent_verification_sha256": _sha256_json(parent_verification),
        "prelock_git_identity_sha256": _sha256_json(prelock_git_identity),
        "output_path": OUTPUT_DIRECTORY.as_posix(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_directory_fsync_supported": parent_fsync_supported,
    }
    if (
        content["git_commit"] != prelock_git_identity.get("upstream_commit")
        or content["input_git_blob"]
        != prelock_git_identity.get("input_head_object_id")
        or parent_verification.get("semantic_evidence_sha256")
        != PARENT_SEMANTIC_EVIDENCE_SHA256
        or parent_verification.get("manifest_self_sha256")
        != PARENT_MANIFEST_SELF_SHA256
    ):
        raise ContextualExpertAggregationAuditError(
            "attempt lock inputs do not match the frozen authorization"
        )
    lock_bytes = _json_bytes(content)
    try:
        _experiment._exclusive_write(lock_path, lock_bytes)
        fsync_supported = _experiment._fsync_directory(parent)
        persisted = lock_path.read_bytes()
    except (OSError, _experiment.ContextualExpertAggregationExperimentError) as exc:
        # Never remove a possibly created lock after an ambiguous failure.
        raise ContextualExpertAggregationAuditError(
            "audit attempt lock could not be durably created"
        ) from exc
    if persisted != lock_bytes:
        raise ContextualExpertAggregationAuditError(
            "audit attempt lock changed during durable readback"
        )
    if fsync_supported != parent_fsync_supported:
        raise ContextualExpertAggregationAuditError(
            "audit lock parent fsync support changed during creation"
        )
    return AuditLockEvidence(
        path=ATTEMPT_LOCK_PATH.as_posix(),
        sha256=_experiment.sha256_bytes(lock_bytes),
        content=content,
        bytes_value=lock_bytes,
        parent_directory_fsync_supported=parent_fsync_supported,
    )


def read_attempt_lock(repo_root: Path) -> AuditLockEvidence:
    root = repo_root.resolve()
    try:
        path = _experiment._harden_path(
            root / ATTEMPT_LOCK_PATH,
            label="Durable audit attempt lock",
            require_exists=True,
        )
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise ContextualExpertAggregationAuditError(
            "audit attempt lock path is redirected or missing"
        ) from exc
    if not path.is_file():
        raise ContextualExpertAggregationAuditError(
            "audit attempt lock is not a regular file"
        )
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationAuditError(
            "audit attempt lock is missing or unreadable"
        ) from exc
    if not isinstance(value, dict) or _json_bytes(value) != payload:
        raise ContextualExpertAggregationAuditError(
            "audit attempt lock is not canonical exact JSON"
        )
    expected_keys = {
        "schema_version",
        "contract_version",
        "stage",
        "run_id",
        "attempt_number",
        "one_run_no_retry",
        "persistent_after_success_or_failure",
        "created_before_through_2023_worktree_read",
        "created_before_any_2019_market_value",
        "expected_branch",
        "git_commit",
        "input_path",
        "input_git_blob",
        "input_raw_sha256",
        "parent_manifest_sha256",
        "parent_checkpoint_payload_sha256",
        "parent_checkpoint_self_sha256",
        "parent_semantic_evidence_sha256",
        "parent_verification_sha256",
        "prelock_git_identity_sha256",
        "output_path",
        "created_at_utc",
        "parent_directory_fsync_supported",
    }
    created_at = value.get("created_at_utc") if isinstance(value, dict) else None
    try:
        parsed_created_at = (
            datetime.fromisoformat(created_at)
            if type(created_at) is str
            else None
        )
    except ValueError:
        parsed_created_at = None
    created_at_valid = (
        parsed_created_at is not None
        and parsed_created_at.tzinfo is not None
        and parsed_created_at.utcoffset() is not None
        and parsed_created_at.utcoffset().total_seconds() == 0.0
        and parsed_created_at.isoformat() == created_at
    )
    if (
        set(value) != expected_keys
        or value["schema_version"] != ATTEMPT_LOCK_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["stage"] != AUDIT_STAGE
        or value["run_id"] != AUDIT_RUN_ID
        or type(value["attempt_number"]) is not int
        or value["attempt_number"] != 1
        or value["one_run_no_retry"] is not True
        or value["persistent_after_success_or_failure"] is not True
        or value["created_before_through_2023_worktree_read"] is not True
        or value["created_before_any_2019_market_value"] is not True
        or value["expected_branch"] != EXPECTED_BRANCH
        or type(value["git_commit"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value["git_commit"])
        is None
        or value["input_path"] != AUDIT_INPUT_PATH.as_posix()
        or value["input_git_blob"] != AUDIT_INPUT_SPEC.git_blob
        or value["input_raw_sha256"] != AUDIT_INPUT_SPEC.raw_sha256
        or value["parent_manifest_sha256"] != PARENT_MANIFEST_SELF_SHA256
        or value["parent_checkpoint_payload_sha256"]
        != PARENT_CHECKPOINT_PAYLOAD_SHA256
        or value["parent_checkpoint_self_sha256"]
        != PARENT_CHECKPOINT_SELF_SHA256
        or value["parent_semantic_evidence_sha256"]
        != PARENT_SEMANTIC_EVIDENCE_SHA256
        or type(value["parent_verification_sha256"]) is not str
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", value["parent_verification_sha256"]
        )
        is None
        or type(value["prelock_git_identity_sha256"]) is not str
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", value["prelock_git_identity_sha256"]
        )
        is None
        or value["output_path"] != OUTPUT_DIRECTORY.as_posix()
        or not created_at_valid
        or type(value["parent_directory_fsync_supported"]) is not bool
    ):
        raise ContextualExpertAggregationAuditError(
            "audit attempt lock content violates the frozen contract"
        )
    return AuditLockEvidence(
        path=ATTEMPT_LOCK_PATH.as_posix(),
        sha256=_experiment.sha256_bytes(payload),
        content=value,
        bytes_value=payload,
        parent_directory_fsync_supported=value[
            "parent_directory_fsync_supported"
        ],
    )


def prepare_rejected_development(
    repo_root: Path, *, deadline: _experiment.StageDeadline
) -> PreparedDevelopment:
    """Regenerate the exact rejected parent using through-2018 data only."""

    from .contextual_expert_aggregation_audit_parent import (
        verify_rejected_parent,
    )

    root = repo_root.resolve()
    deadline.check("before rejected-parent verification")
    evidence = verify_rejected_parent(root)
    deadline.check("after rejected-parent verification")
    parent = evidence.bundle
    snapshot = _experiment.load_authorized_prices(root, stage="development")
    online = _replay.replay_from_empty(snapshot.frame)
    checkpoint_payload = _v2_stage._bundle_payload(
        parent, _experiment.DEVELOPMENT_CHECKPOINT_FILENAME
    )
    checkpoint = _v2_artifacts.parse_composite_stage_checkpoint_bytes(
        checkpoint_payload
    )
    payload_sha = _v2_stage._require_authorized_checkpoint_payload(
        checkpoint_payload,
        checkpoint,
        authorized_payload_sha256=PARENT_CHECKPOINT_PAYLOAD_SHA256,
    )
    if (
        payload_sha != PARENT_CHECKPOINT_PAYLOAD_SHA256
        or checkpoint.checkpoint_sha256 != PARENT_CHECKPOINT_SELF_SHA256
    ):
        raise ContextualExpertAggregationAuditError(
            "rejected-parent checkpoint identity changed"
        )
    for arm in _v2_artifacts.ARM_ORDER:
        if checkpoint.replay_checkpoints[arm].to_dict() != online.checkpoint.to_dict():
            raise ContextualExpertAggregationAuditError(
                f"rejected-parent checkpoint does not replay for {arm}"
            )
    replay_artifacts = _v2_stage._require_development_replay_artifacts(
        parent, regenerated=online
    )
    if (
        _experiment.sha256_bytes(evidence.manifest_bytes)
        != evidence.manifest_file_sha256
        or _experiment.sha256_bytes(evidence.checksums_bytes)
        != evidence.checksums_file_sha256
    ):
        raise ContextualExpertAggregationAuditError(
            "rejected-parent metadata changed after verification"
        )
    deadline.check("after through-2018 checkpoint replay")
    return PreparedDevelopment(
        parent_bundle=parent,
        parent_verification=evidence.lock_payload(),
        snapshot=snapshot,
        online_replay=online,
        checkpoint=checkpoint,
        parent_manifest_bytes=evidence.manifest_bytes,
        parent_checksums_bytes=evidence.checksums_bytes,
        parent_replay_artifacts=replay_artifacts,
    )


def load_locked_audit_snapshot(
    repo_root: Path,
    *,
    development: PreparedDevelopment,
    prelock_git_identity: Mapping[str, Any],
    lock: AuditLockEvidence,
    deadline: _experiment.StageDeadline,
    allowed_output_directories: Sequence[str] = (),
) -> tuple[_experiment.LoadedPriceSnapshot, Mapping[str, Any]]:
    """First authorized input read; return only after lineage and prefix proof."""

    root = repo_root.resolve()
    deadline.check("before durable audit lock readback")
    persisted = read_attempt_lock(root)
    if (
        persisted.sha256 != lock.sha256
        or persisted.bytes_value != lock.bytes_value
        or persisted.content != dict(lock.content)
    ):
        raise ContextualExpertAggregationAuditError(
            "audit lock changed before the first input read"
        )
    deadline.check("immediately before first audit input worktree read")
    tracked = require_postlock_input_identity(
        root,
        prelock_git_identity,
        allowed_output_directories=allowed_output_directories,
    )
    deadline.check("after exact audit input worktree identity")
    _experiment.verify_authorized_price_lineage(root, AUDIT_INPUT_SPEC)
    deadline.check("before bounded audit input parsing")
    raw = _experiment.load_bounded_price_snapshot(
        root / AUDIT_INPUT_PATH, spec=AUDIT_INPUT_SPEC
    )
    deadline.check("after bounded audit input parsing")
    provenance = {
        **dict(raw.provenance),
        "tracked_input": {
            "path": tracked["path"],
            "sha256": tracked["sha256"],
            "git_blob": tracked["git_blob"],
        },
        "authorized_loader_lineage": {
            "source_manifest_sha256": (
                None
                if AUDIT_INPUT_SPEC.source_bundle is None
                else AUDIT_INPUT_SPEC.source_bundle.manifest_self_sha256
            ),
            "source_provenance_sha256": AUDIT_INPUT_SPEC.source_provenance_sha256,
            "source_parent_manifest_sha256": (
                None
                if AUDIT_INPUT_SPEC.source_parent_bundle is None
                else AUDIT_INPUT_SPEC.source_parent_bundle.manifest_self_sha256
            ),
        },
    }
    snapshot = _experiment._attest_loaded_snapshot(
        _experiment.LoadedPriceSnapshot(
            spec=raw.spec,
            frame=raw.frame,
            raw_sha256=raw.raw_sha256,
            canonical_csv_bytes=raw.canonical_csv_bytes,
            provenance=provenance,
        ),
        authorized_lineage=True,
    )
    _experiment.require_loaded_snapshot_integrity(
        snapshot, expected_spec=AUDIT_INPUT_SPEC
    )
    _experiment.require_snapshot_prefix(
        development.snapshot,
        snapshot,
        development_path=root / DEVELOPMENT_INPUT_PATH,
        confirmation_path=root / AUDIT_INPUT_PATH,
    )
    # Reproduce the full through-2018 model prefix once more before any caller
    # receives the object that contains 2019-2023 values.
    prefix_frame = snapshot.frame.loc[
        snapshot.frame.index <= pd.Timestamp("2018-12-31")
    ].copy()
    if not prefix_frame.equals(development.snapshot.frame):
        raise ContextualExpertAggregationAuditError(
            "audit input through-2018 canonical frame changed"
        )
    prefix_replay = _replay.replay_from_empty(prefix_frame)
    if prefix_replay.checkpoint.to_dict() != development.online_replay.checkpoint.to_dict():
        raise ContextualExpertAggregationAuditError(
            "audit input through-2018 model checkpoint changed"
        )
    deadline.check("after locked input lineage and through-2018 prefix proof")
    audit_provenance = {
        "authorized_snapshot": dict(snapshot.provenance),
        "postlock_tracked_input": dict(tracked),
        "attempt_lock_path": lock.path,
        "attempt_lock_sha256": lock.sha256,
        "input_opened_only_after_durable_lock": True,
        "through_2018_raw_and_canonical_prefix_exact": True,
        "through_2018_model_checkpoint_exact": True,
        "post_2023_market_values_accessed": False,
    }
    audit_provenance["audit_provenance_sha256"] = _sha256_json(audit_provenance)
    return snapshot, audit_provenance


def compute_audit_continuation(
    repo_root: Path,
    *,
    development: PreparedDevelopment,
    snapshot: _experiment.LoadedPriceSnapshot,
    input_provenance: Mapping[str, Any],
    deadline: _experiment.StageDeadline,
) -> AuditComputation:
    """Continue all exact v2 arms and all continuous ledgers through 2023."""

    root = repo_root.resolve()
    deadline.check("audit continuation entry")
    suffix = snapshot.frame.loc[
        snapshot.frame.index
        > pd.Timestamp(development.online_replay.checkpoint.checkpoint_date)
    ].copy()
    if (
        suffix.empty
        or suffix.index[0] < AUDIT_SUFFIX_START
        or suffix.index[-1] > AUDIT_SUFFIX_END
        or suffix.index[-1].date().isoformat() != LAST_OBSERVED_SESSION
    ):
        raise ContextualExpertAggregationAuditError(
            "audit continuation suffix has the wrong frozen boundary"
        )
    forks = _replay.fork_confirmation_arms(
        suffix,
        development.online_replay.checkpoint,
        historical_prefix=development.snapshot.frame,
    )
    deadline.check("after four-arm causal continuation replay")
    replays = dict(forks.arms)
    frozen_state = replays[_replay.FROZEN_2018_ARM].checkpoint.model_payload[
        "state"
    ]
    development_state = development.online_replay.checkpoint.model_payload[
        "state"
    ]
    if (
        replays[_replay.FROZEN_2018_ARM].diagnostics.admitted_events != 0
        or frozen_state["admitted_lesson_count"]
        != development_state["admitted_lesson_count"]
    ):
        raise ContextualExpertAggregationAuditError(
            "frozen-2018 arm admitted a post-cutoff lesson"
        )
    full_online = _replay.replay_from_empty(snapshot.frame)
    resume = _replay.verify_resume_equivalence(
        full_online, replays[_replay.ONLINE_FULL_ARM]
    )
    resume.require()
    deadline.check("after full replay resume-equivalence proof")
    fixed = _v2_stage._fixed_features_for_replays(replays)
    parent_proof, source_provenance = _v2_stage._parent_causal_prefix_proof(
        root,
        stage=_v2_artifacts.CONFIRMATION_STAGE,
        fixed_features=fixed,
    )
    development_replays = {
        arm: development.online_replay for arm in _v2_artifacts.ARM_ORDER
    }
    ledgers, accounts = _v2_stage._run_confirmation_ledgers(
        development_bundle=development.parent_bundle,
        development_snapshot=development.snapshot,
        confirmation_snapshot=snapshot,
        development_replays=development_replays,
        confirmation_replays=replays,
        development_checkpoint=development.checkpoint,
    )
    deadline.check("after continuous audit ledgers")
    episodes, xors = _v2_stage._extract_episodes_and_xors(ledgers)
    deadline.check("after audit replay and continuous ledgers")

    full_forecasts: dict[str, pd.DataFrame] = {}
    full_matured: dict[str, pd.DataFrame] = {}
    for arm in _v2_artifacts.ARM_ORDER:
        full_forecasts[arm] = pd.concat(
            [development.online_replay.forecast, replays[arm].forecast]
        )
        full_matured[arm] = pd.concat(
            [
                development.online_replay.matured_lessons,
                replays[arm].matured_lessons,
            ],
            ignore_index=True,
        )
        if (
            not full_forecasts[arm].index.is_unique
            or len(full_forecasts[arm]) != len(snapshot.frame)
            or not full_forecasts[arm].index.equals(snapshot.frame.index)
        ):
            raise ContextualExpertAggregationAuditError(
                f"continuous audit forecast boundary changed for {arm}"
            )
    full_fixed = pd.concat(
        [development.online_replay.opportunity_frame, fixed]
    )
    if not full_fixed.index.equals(snapshot.frame.index):
        raise ContextualExpertAggregationAuditError(
            "continuous audit fixed-feature boundary changed"
        )

    replay_diagnostics = {
        "audit_replay_diagnostics_schema_version": 1,
        "stage": AUDIT_STAGE,
        "v2_continuation_runtime_semantics": "confirmation_style_fork_only",
        "arm_order": list(_v2_artifacts.ARM_ORDER),
        "by_arm": {
            arm: {
                "diagnostics": _jsonable(asdict(replays[arm].diagnostics)),
                "checkpoint_sha256": replays[arm].checkpoint.digest_sha256,
            }
            for arm in _v2_artifacts.ARM_ORDER
        },
        "common_fork_checkpoint_sha256": forks.checkpoint_sha256,
        "online_resume_equivalence": _jsonable(asdict(resume)),
        "frozen_post_cutoff_admitted_events": 0,
        "frozen_terminal_admitted_count_equals_2018": True,
    }
    replay_diagnostics["diagnostics_sha256"] = _sha256_json(replay_diagnostics)
    prefix_proof = {
        "audit_prefix_proof_schema_version": 1,
        "stage": AUDIT_STAGE,
        "parent_manifest_sha256": PARENT_MANIFEST_SELF_SHA256,
        "parent_checkpoint_payload_sha256": PARENT_CHECKPOINT_PAYLOAD_SHA256,
        "parent_checkpoint_self_sha256": PARENT_CHECKPOINT_SELF_SHA256,
        "parent_semantic_evidence_sha256": PARENT_SEMANTIC_EVIDENCE_SHA256,
        "fork_parent_checkpoint_sha256": forks.checkpoint_sha256,
        "online_resume_equivalence": _jsonable(asdict(resume)),
        "development_parent_replay_artifacts": _jsonable(
            development.parent_replay_artifacts
        ),
        "development_account_prefix_verified_for_every_cost_policy": True,
        "audit_input_provenance_sha256": input_provenance[
            "audit_provenance_sha256"
        ],
        "suffix_first_session": suffix.index[0].date().isoformat(),
        "suffix_last_session": suffix.index[-1].date().isoformat(),
        "account_reset_count_after_inception": 0,
    }
    prefix_proof["proof_sha256"] = _sha256_json(prefix_proof)
    audit_parent_proof = {
        "stage": AUDIT_STAGE,
        "v2_confirmation_style_source_proof": parent_proof,
        "source_provenance": source_provenance,
    }
    audit_parent_proof["proof_sha256"] = _sha256_json(audit_parent_proof)
    return AuditComputation(
        snapshot=snapshot,
        replays=replays,
        full_forecasts=full_forecasts,
        full_matured_lessons=full_matured,
        full_fixed_features=full_fixed,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        parent_causal_prefix_proof=audit_parent_proof,
        source_bundle_provenance=source_provenance,
        replay_diagnostics=replay_diagnostics,
        prefix_continuity_proof=prefix_proof,
    )


def _evaluation_evidence(
    computation: AuditComputation,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, pd.DataFrame]]:
    from . import contextual_expert_aggregation_audit_evaluation as _audit_eval

    primary: dict[str, Any] = {}
    comparators: dict[str, Any] = {}
    for cost in _v2_artifacts.COST_ORDER:
        benchmark = computation.ledgers[cost]["aapl_buy_hold"]
        primary[cost] = {
            "strategy_ledger": computation.ledgers[cost][_replay.ONLINE_FULL_ARM],
            "benchmark_ledger": benchmark,
            "complete_episodes": computation.episodes[cost][
                _replay.ONLINE_FULL_ARM
            ].complete,
        }
        comparators[cost] = {}
        for name in _audit_eval.AUDIT_COMPARATOR_NAMES:
            policy = (
                name
                if name in _v2_artifacts.FIXED_POLICY_ORDER
                else _v2_stage.ABLATION_POLICY_BY_COMPARISON[name]
            )
            comparators[cost][name] = {
                "strategy_ledger": computation.ledgers[cost][policy],
                "benchmark_ledger": benchmark,
                "complete_episodes": computation.episodes[cost][policy].complete,
                "learner_minus_comparator_xor": computation.xors[cost][name].complete,
            }

    online = computation.full_forecasts[_replay.ONLINE_FULL_ARM]
    adaptive: dict[str, pd.DataFrame] = {}
    ledger_dates = computation.ledgers[_v2_artifacts.COST_ORDER[0]][
        _replay.ONLINE_FULL_ARM
    ]["fill_date"].tolist()
    decision_index = pd.DatetimeIndex(ledger_dates)
    for name in _audit_eval.ADAPTIVE_COMPARISON_NAMES:
        policy = _v2_stage.ABLATION_POLICY_BY_COMPARISON[name]
        comparator = computation.full_forecasts[policy]
        frame = pd.DataFrame(
            {
                "decision_date": [item.date().isoformat() for item in decision_index],
                "canonical_union_opportunity": online.loc[
                    decision_index, "canonical_union_opportunity"
                ].astype(bool).tolist(),
                "online_cash_score": online.loc[
                    decision_index, "cash_score"
                ].astype(float).tolist(),
                "comparator_cash_score": comparator.loc[
                    decision_index, "cash_score"
                ].astype(float).tolist(),
                "online_action": online.loc[decision_index, "action"].tolist(),
                "comparator_action": comparator.loc[
                    decision_index, "action"
                ].tolist(),
            },
            columns=list(_audit_eval.ADAPTIVE_STATE_COLUMNS),
        )
        adaptive[name] = frame
    return primary, comparators, adaptive


def evaluate_audit(
    computation: AuditComputation,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    """Reconcile all ledgers, apply audit gates, and preserve fatal failures."""

    from . import contextual_expert_aggregation_audit_evaluation as _audit_eval

    stage_view = _v2_stage.StageComputation(
        stage=_v2_artifacts.CONFIRMATION_STAGE,
        snapshot=computation.snapshot,
        replays=computation.replays,
        ledgers=computation.ledgers,
        accounts=computation.accounts,
        episodes=computation.episodes,
        xors=computation.xors,
        parent_causal_prefix_proof=computation.parent_causal_prefix_proof,
        source_bundle_provenance=computation.source_bundle_provenance,
        replay_diagnostics=computation.replay_diagnostics,
        prefix_continuity_proof=computation.prefix_continuity_proof,
        confirmation_authorization=None,
        development_parent_manifest_bytes=None,
        development_parent_checksums_bytes=None,
    )
    fatal_reasons = _v2_stage.classify_terminal_residuals(
        computation.episodes,
        computation.xors,
        stage=_v2_artifacts.CONFIRMATION_STAGE,
    )
    integrity = {name: True for name in AUDIT_INTEGRITY_CHECKS}
    if fatal_reasons:
        integrity["ledger_episode_and_xor_reconciliation_exact"] = False
    if fatal_reasons:
        checks = {
            **{f"integrity.{name}": bool(value) for name, value in integrity.items()},
            "economic.post_rejection_2019_2023_pass": False,
            "economic.continuous_2005_2023_robustness_pass": False,
            "economic.historical_policy_candidate_for_2024_audit": False,
        }
        failures = [name for name, value in checks.items() if not value]
        gate = {
            "audit_gate_schema_version": 1,
            "stage": AUDIT_STAGE,
            "passed": False,
            "checks": checks,
            "failed_checks": failures,
            "fatal_integrity_rejection": True,
            "fatal_reasons": _jsonable(fatal_reasons),
            "normal_economic_gates_evaluated": False,
        }
        metrics = {
            "audit_metrics_schema_version": 1,
            "stage": AUDIT_STAGE,
            "status": "not_scored_due_to_fatal_terminal_integrity",
            "fatal_reasons": _jsonable(fatal_reasons),
        }
        return integrity, gate, metrics

    _v2_stage._reconcile_evidence(stage_view)
    primary, comparators, adaptive = _evaluation_evidence(computation)
    metrics = _audit_eval.apply_audit_gates(
        primary,
        comparator_evidence=comparators,
        adaptive_state_differences=adaptive,
    )
    statuses = metrics["decision_statuses"]
    checks: dict[str, bool] = {
        **{f"integrity.{name}": bool(value) for name, value in integrity.items()},
        "economic.post_rejection_2019_2023_pass": bool(
            statuses["post_rejection_2019_2023_pass"]
        ),
        "economic.continuous_2005_2023_robustness_pass": bool(
            statuses["continuous_2005_2023_robustness_pass"]
        ),
        "economic.historical_policy_candidate_for_2024_audit": bool(
            statuses["historical_policy_candidate_for_2024_audit"]
        ),
    }
    for cost, report in metrics["policy_by_cost"].items():
        for period_name in (
            "suffix_2019_2023",
            "continuous_2005_2023",
        ):
            criterion = report[period_name]["criterion"]
            for name, value in criterion["checks"].items():
                if value is not None:
                    checks[f"{cost}.{period_name}.{name}"] = bool(value)
    failures = [name for name, value in checks.items() if not value]
    gate = {
        "audit_gate_schema_version": 1,
        "stage": AUDIT_STAGE,
        "passed": bool(statuses["historical_policy_candidate_for_2024_audit"]),
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "checks": checks,
        "failed_checks": failures,
        "fatal_integrity_rejection": False,
        "fatal_reasons": [],
        "normal_economic_gates_evaluated": True,
        "learning_candidate_for_2024_audit": bool(
            statuses["learning_candidate_for_2024_audit"]
        ),
        "original_v2_rejection_remains_final": True,
    }
    return integrity, gate, {
        "audit_metrics_schema_version": 1,
        "stage": AUDIT_STAGE,
        "status": "scored",
        **metrics,
    }


def _diagnostic_frame(computation: AuditComputation) -> pd.DataFrame:
    selected = [
        name
        for name in _v2_stage.DIAGNOSTIC_COLUMNS
        if name not in {"arm", "decision_date"}
    ]
    frames: list[pd.DataFrame] = []
    for arm in _v2_artifacts.ARM_ORDER:
        value = computation.full_forecasts[arm].loc[:, selected].copy()
        value.insert(
            0,
            "decision_date",
            value.index.map(lambda item: item.date().isoformat()),
        )
        value.insert(0, "arm", arm)
        frames.append(
            value.reset_index(drop=True).loc[
                :, list(_v2_stage.DIAGNOSTIC_COLUMNS)
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _episodes_payload(
    computation: AuditComputation, *, cost: str
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy in _v2_artifacts.POLICY_ORDER:
        extraction = computation.episodes[cost][policy]
        policies[policy] = {
            "complete_columns": list(_ledger.EPISODE_COLUMNS),
            "complete": _jsonable(extraction.complete.to_dict(orient="records")),
            "unresolved_columns": list(_ledger.UNRESOLVED_EPISODE_COLUMNS),
            "unresolved": _jsonable(
                extraction.unresolved.to_dict(orient="records")
            ),
        }
    value = {
        "audit_episode_artifact_schema_version": 1,
        "stage": AUDIT_STAGE,
        "cost": cost,
        "cost_bps": _v2_artifacts.COST_BPS[cost],
        "policy_order": list(_v2_artifacts.POLICY_ORDER),
        "policies": policies,
    }
    value["artifact_sha256"] = _sha256_json(value)
    return value


def _xors_payload(
    computation: AuditComputation, *, cost: str
) -> dict[str, Any]:
    comparisons = _v2_stage._active_comparison_order(
        _v2_artifacts.CONFIRMATION_STAGE
    )
    values: dict[str, Any] = {}
    for name in comparisons:
        extraction = computation.xors[cost][name]
        values[name] = {
            "complete_columns": list(_ledger.XOR_COLUMNS),
            "complete": _jsonable(extraction.complete.to_dict(orient="records")),
            "unresolved_columns": list(_ledger.UNRESOLVED_XOR_COLUMNS),
            "unresolved": _jsonable(
                extraction.unresolved.to_dict(orient="records")
            ),
        }
    value = {
        "audit_xor_artifact_schema_version": 1,
        "stage": AUDIT_STAGE,
        "cost": cost,
        "cost_bps": _v2_artifacts.COST_BPS[cost],
        "comparison_order": list(comparisons),
        "comparisons": values,
    }
    value["artifact_sha256"] = _sha256_json(value)
    return value


def _pending_lessons_payload(computation: AuditComputation) -> dict[str, Any]:
    value = {
        "audit_pending_artifact_schema_version": 1,
        "stage": AUDIT_STAGE,
        "arm_order": list(_v2_artifacts.ARM_ORDER),
        "by_arm": {
            arm: _jsonable(
                list(computation.replays[arm].checkpoint.pending_lessons)
            )
            for arm in _v2_artifacts.ARM_ORDER
        },
    }
    value["artifact_sha256"] = _sha256_json(value)
    return value


def _integrity_payload(
    *,
    integrity: Mapping[str, bool],
    computation: AuditComputation,
    lock: AuditLockEvidence,
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if set(integrity) != set(AUDIT_INTEGRITY_CHECKS):
        raise ContextualExpertAggregationAuditError(
            "audit integrity inventory changed"
        )
    value = {
        "audit_integrity_evidence_schema_version": 1,
        "stage": AUDIT_STAGE,
        "checks": {name: bool(integrity[name]) for name in sorted(integrity)},
        "attempt_lock": {
            "path": lock.path,
            "sha256": lock.sha256,
            "parent_directory_fsync_supported": (
                lock.parent_directory_fsync_supported
            ),
        },
        "input_provenance_sha256": input_provenance["audit_provenance_sha256"],
        "parent_causal_prefix_proof_sha256": computation.parent_causal_prefix_proof[
            "proof_sha256"
        ],
        "prefix_continuity_proof_sha256": computation.prefix_continuity_proof[
            "proof_sha256"
        ],
        "replay_diagnostics_sha256": computation.replay_diagnostics[
            "diagnostics_sha256"
        ],
        "zero_use": {
            "network_access": False,
            "news_access": False,
            "llm_calls": 0,
            "api_calls": 0,
            "external_model_calls": 0,
            "external_cost_usd": 0.0,
        },
    }
    value["integrity_evidence_sha256"] = _sha256_json(value)
    return value


def _runtime_evidence(
    deadline: _experiment.StageDeadline, *, phase: str
) -> dict[str, Any]:
    elapsed = deadline.check(phase)
    return {
        "runtime_evidence_schema_version": 1,
        "stage": AUDIT_STAGE,
        "seconds_before_seal": elapsed,
        "strict_limit_seconds": RUN_TIME_LIMIT_SECONDS,
        "within_limit_before_seal": elapsed < RUN_TIME_LIMIT_SECONDS,
        "monotonic_clock": True,
        "strict_limit_rechecked_immediately_before_promotion": True,
        "network_access": False,
        "news_access": False,
        "llm_calls": 0,
        "api_calls": 0,
        "external_model_calls": 0,
        "external_cost_usd": 0.0,
    }


def _report(
    *,
    gate_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    lock: AuditLockEvidence,
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    passed = bool(gate_report["passed"])
    statuses = (
        metrics.get("decision_statuses", {})
        if metrics.get("status") == "scored"
        else {}
    )
    value = {
        "audit_report_schema_version": 1,
        "contract_version": CONTRACT_VERSION,
        "stage": AUDIT_STAGE,
        "run_id": AUDIT_RUN_ID,
        "status": (
            "POST_REJECTION_AUDIT_POLICY_CANDIDATE"
            if passed
            else "POST_REJECTION_AUDIT_REJECTED"
        ),
        "stage_pass": passed,
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "is_confirmation": False,
        "is_holdout_test": False,
        "is_prospective": False,
        "parent_rejection_remains_final": True,
        "historical_results_authorize_real_capital": False,
        "calendar_cutoff": "2023-12-31",
        "physical_data_end": LAST_OBSERVED_SESSION,
        "post_2023_market_values_accessed": False,
        "continuous_account_start": ACCOUNT_START,
        "account_reset_count_after_inception": 0,
        "attempt_lock_path": lock.path,
        "attempt_lock_sha256": lock.sha256,
        "input_provenance_sha256": input_provenance["audit_provenance_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "historical_policy_candidate_for_2024_audit": bool(
            statuses.get("historical_policy_candidate_for_2024_audit", False)
        ),
        "learning_candidate_for_2024_audit": bool(
            statuses.get("learning_candidate_for_2024_audit", False)
        ),
        "later_data_access_authorized": False,
        "gate_report": _jsonable(gate_report),
        "metrics": _jsonable(metrics),
        "runtime_cost_evidence": _jsonable(runtime),
        "next_step": (
            "A separate preregistered 2024+ audit may be proposed; no later "
            "data is authorized by this result."
            if passed
            else "This exact v2 continuation is permanently closed."
        ),
    }
    value["report_sha256"] = _sha256_json(value)
    return value


def build_audit_payloads(
    *,
    computation: AuditComputation,
    development: PreparedDevelopment,
    parent_verification: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
    lock: AuditLockEvidence,
    integrity: Mapping[str, bool],
    gate_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any], dict[str, Any]]:
    checkpoint = build_audit_checkpoint(
        replay_checkpoints={
            arm: computation.replays[arm].checkpoint
            for arm in _v2_artifacts.ARM_ORDER
        },
        administrative_accounts=computation.accounts,
    )
    report = _report(
        gate_report=gate_report,
        metrics=metrics,
        runtime=runtime,
        checkpoint=checkpoint,
        lock=lock,
        input_provenance=input_provenance,
    )
    payloads: dict[str, bytes] = {
        ".gitattributes": GIT_ATTRIBUTES_BYTES,
        ATTEMPT_LOCK_FILENAME: lock.bytes_value,
        "rejected_parent_manifest.json": development.parent_manifest_bytes,
        "rejected_parent_checksums.json": development.parent_checksums_bytes,
        "rejected_parent_verification.json": _json_bytes(parent_verification),
        "input_provenance.json": _json_bytes(input_provenance),
        "audit_prices_through_2023.csv": computation.snapshot.canonical_csv_bytes,
        "audit_fixed_features.table.json": _v2_stage._decision_table_bytes(
            computation.full_fixed_features,
            schema=_v2_stage.FIXED_FEATURE_SCHEMA,
        ),
        "audit_forecast__fixed_comparators.table.json": (
            _v2_stage._decision_table_bytes(
                computation.full_fixed_features.loc[
                    :, list(_replay.COMPARATOR_TARGET_COLUMNS)
                ],
                schema=_v2_stage.FIXED_COMPARATOR_SCHEMA,
            )
        ),
        "audit_state_weight_diagnostics.table.json": _v2_stage.table_bytes(
            _diagnostic_frame(computation), schema=_v2_stage.DIAGNOSTIC_SCHEMA
        ),
        "audit_replay_diagnostics.json": _json_bytes(
            computation.replay_diagnostics
        ),
        "audit_pending_lessons.json": _json_bytes(
            _pending_lessons_payload(computation)
        ),
        "audit_continuation_checkpoint_through_2023.json": (
            audit_checkpoint_bytes(checkpoint)
        ),
        "audit_metrics.json": _json_bytes(metrics),
        "audit_integrity_evidence.json": _json_bytes(
            _integrity_payload(
                integrity=integrity,
                computation=computation,
                lock=lock,
                input_provenance=input_provenance,
            )
        ),
        "audit_gate_report.json": _json_bytes(gate_report),
        "audit_parent_causal_prefix_proof.json": _json_bytes(
            computation.parent_causal_prefix_proof
        ),
        "audit_prefix_continuity_proof.json": _json_bytes(
            computation.prefix_continuity_proof
        ),
        "source_bundle_provenance.json": _json_bytes(
            computation.source_bundle_provenance
        ),
        "audit_runtime_cost_evidence.json": _json_bytes(runtime),
        "report.json": _json_bytes(report),
    }
    for arm in _v2_artifacts.ARM_ORDER:
        payloads[f"audit_forecast__{arm}.table.json"] = (
            _v2_stage._decision_table_bytes(
                computation.full_forecasts[arm], schema=_v2_stage.FORECAST_SCHEMA
            )
        )
        payloads[f"audit_matured_lessons__{arm}.table.json"] = (
            _v2_stage.table_bytes(
                computation.full_matured_lessons[arm],
                schema=_v2_stage.MATURED_EVENT_SCHEMA,
            )
        )
    for cost in _v2_artifacts.COST_ORDER:
        for policy in _v2_artifacts.POLICY_ORDER:
            payloads[f"audit_ledger__{cost}__{policy}.table.json"] = (
                _v2_stage.ledger_table_bytes(
                    computation.ledgers[cost][policy]
                )
            )
        payloads[f"audit_episodes__{cost}.json"] = _json_bytes(
            _episodes_payload(computation, cost=cost)
        )
        payloads[f"audit_xor__{cost}.json"] = _json_bytes(
            _xors_payload(computation, cost=cost)
        )
    if set(payloads) != set(AUDIT_PAYLOAD_NAMES):
        raise ContextualExpertAggregationAuditError(
            "constructed audit payload inventory changed"
        )
    return payloads, checkpoint, report


def run_audit(repo_root: Path | None = None) -> dict[str, Any]:
    """Execute and seal the one authorized audit attempt."""

    from .contextual_expert_aggregation_audit_bootstrap import (
        require_active_attestation,
    )

    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    require_active_attestation(operation="stage", stage=AUDIT_STAGE, repo_root=root)
    deadline = _experiment.StageDeadline(limit_seconds=RUN_TIME_LIMIT_SECONDS)
    deadline.check("audit runner entry")
    require_unused_destination(root)
    deadline.check("after unused-destination proof")
    prelock = require_prelock_git_identity(root)
    deadline.check("after first pre-lock Git proof")
    development = prepare_rejected_development(root, deadline=deadline)
    repeated_prelock = require_prelock_git_identity(root)
    if repeated_prelock != prelock:
        raise ContextualExpertAggregationAuditError(
            "pre-lock Git identity changed during rejected-parent verification"
        )
    deadline.check("immediately before irreversible audit attempt lock")
    lock = create_attempt_lock(
        repo_root=root,
        prelock_git_identity=prelock,
        parent_verification=development.parent_verification,
    )
    deadline.check("after irreversible audit attempt lock")
    snapshot, input_provenance = load_locked_audit_snapshot(
        root,
        development=development,
        prelock_git_identity=prelock,
        lock=lock,
        deadline=deadline,
    )
    computation = compute_audit_continuation(
        root,
        development=development,
        snapshot=snapshot,
        input_provenance=input_provenance,
        deadline=deadline,
    )
    deadline.check("before audit evaluation")
    integrity, gate_report, metrics = evaluate_audit(computation)
    deadline.check("after audit evaluation")
    runtime = _runtime_evidence(deadline, phase="before audit payload construction")
    payloads, checkpoint, report = build_audit_payloads(
        computation=computation,
        development=development,
        parent_verification=development.parent_verification,
        input_provenance=input_provenance,
        lock=lock,
        integrity=integrity,
        gate_report=gate_report,
        metrics=metrics,
        runtime=runtime,
    )
    deadline.check("after audit payload construction")
    manifest_fields = {
        "stage": AUDIT_STAGE,
        "stage_pass": bool(gate_report["passed"]),
        "run_id": AUDIT_RUN_ID,
        "evidence_classification": EVIDENCE_CLASSIFICATION,
        "git_identity": prelock,
        "parent_manifest_sha256": PARENT_MANIFEST_SELF_SHA256,
        "parent_semantic_evidence_sha256": PARENT_SEMANTIC_EVIDENCE_SHA256,
        "input_raw_sha256": AUDIT_INPUT_SPEC.raw_sha256,
        "input_canonical_sha256": AUDIT_INPUT_SPEC.canonical_sha256,
        "attempt_lock_sha256": lock.sha256,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "report_status": report["status"],
        "historical_policy_candidate_for_2024_audit": report[
            "historical_policy_candidate_for_2024_audit"
        ],
        "learning_candidate_for_2024_audit": report[
            "learning_candidate_for_2024_audit"
        ],
        "parent_rejection_remains_final": True,
        "post_2023_market_values_accessed": False,
        "runtime_cost_evidence": runtime,
    }

    def verify_before_promote(
        temporary: Path, manifest: Mapping[str, Any]
    ) -> None:
        from .contextual_expert_aggregation_audit_verifier import (
            verify_private_audit_bundle,
        )

        verify_private_audit_bundle(
            temporary,
            repo_root=root,
            expected_manifest_sha256=str(manifest["manifest_sha256"]),
            deadline=deadline,
        )

    sealed = seal_audit_bundle(
        root / OUTPUT_DIRECTORY,
        manifest_fields=manifest_fields,
        payloads=payloads,
        deadline=deadline,
        before_promote=verify_before_promote,
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "stage": AUDIT_STAGE,
        "run_id": AUDIT_RUN_ID,
        "stage_pass": bool(sealed.manifest["stage_pass"]),
        "status": report["status"],
        "manifest_sha256": sealed.manifest["manifest_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "output_directory": str(sealed.directory),
        "attempt_lock_path": str(root / ATTEMPT_LOCK_PATH),
        "attempt_lock_sha256": lock.sha256,
        "parent_rejection_remains_final": True,
        "post_2023_market_values_accessed": False,
    }


def run_stage(stage: str) -> dict[str, Any]:
    if stage != AUDIT_STAGE:
        raise ContextualExpertAggregationAuditError(
            "the audit runner supports only the frozen audit stage"
        )
    return run_audit()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[AUDIT_STAGE])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_stage(args.stage)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["stage_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AUDIT_INTEGRITY_CHECKS",
    "AuditComputation",
    "AuditLockEvidence",
    "ContextualExpertAggregationAuditError",
    "PreparedDevelopment",
    "build_audit_payloads",
    "compute_audit_continuation",
    "create_attempt_lock",
    "evaluate_audit",
    "load_locked_audit_snapshot",
    "main",
    "prepare_rejected_development",
    "read_attempt_lock",
    "require_postlock_input_identity",
    "require_prelock_git_identity",
    "require_unused_destination",
    "run_audit",
    "run_stage",
]
