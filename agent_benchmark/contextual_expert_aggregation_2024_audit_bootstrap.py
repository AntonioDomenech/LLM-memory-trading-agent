"""Stdlib-only isolated bootstrap for the preregistered 2024 audit.

The only supported commands are exactly::

    python -I -B \
        agent_benchmark/contextual_expert_aggregation_2024_audit_bootstrap.py \
        stage audit_2024
    python -I -B \
        agent_benchmark/contextual_expert_aggregation_2024_audit_bootstrap.py \
        verify audit_2024

No project package is imported at module load time.  Before dispatch, this
module hardens every literal audit path, attests only the frozen dependency
allowlist with exact-path Git commands, and proves the bounded input and
receipt HEAD/index object IDs without opening their worktree bytes.  The
repository data/control tree and every unlisted agent module remain blocked as
import surfaces during the fixed runner or verifier import.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


ATTESTATION_SCHEMA_VERSION = 1
BOOTSTRAP_ID = "contextual-expert-aggregation-2024-audit-isolated-bootstrap-v2"
ALLOWED_OPERATIONS = ("stage", "verify")
ALLOWED_STAGES = ("audit_2024",)
STAGE_DEADLINE_SECONDS = 1_800.0
BOOTSTRAP_DEADLINE_PHASES = (
    "preseal",
    "post_private_verify",
    "prepromotion",
    "final_commit_sample",
    "preverify",
    "postreconstruction",
    "bootstrap_exit",
)

_AGENT_PACKAGE = "agent_benchmark"
_BOOTSTRAP_RELATIVE_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_2024_audit_bootstrap.py"
)
_ACTIVE_ATTESTATION_ATTRIBUTE = (
    "_contextual_expert_aggregation_2024_audit_bootstrap_attestation_v2"
)
_EXPECTED_BRANCH = (
    "codex/aapl-causal-contextual-expert-aggregation-2024-audit-v2"
)
_PREREGISTRATION_COMMIT = "2f4537440eccb41c62ab553d2ffb76460296cc9d"
_EXPECTED_ORIGIN_REPOSITORY = (
    "github.com/AntonioDomenech/LLM-memory-trading-agent"
)
_EXPECTED_ORIGIN_URLS = frozenset(
    {
        "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git",
        "git@github.com:AntonioDomenech/LLM-memory-trading-agent.git",
        "ssh://git@github.com/AntonioDomenech/LLM-memory-trading-agent.git",
    }
)

_CONTROL_ROOT = Path(
    "e/aapl_causal_contextual_expert_aggregation_2024_audit_v2"
)
_AUTHORIZED_INPUTS = _CONTROL_ROOT / "authorized_inputs"
_INPUT_RELATIVE_PATH = _AUTHORIZED_INPUTS / "aapl_spy_qqq_through_2024.csv"
_RECEIPT_RELATIVE_PATH = _CONTROL_ROOT / "input_snapshot_receipt.json"
_INPUT_GIT_BLOB = "3cdb68aac6a3b7b1cda4f64ee81193690f28a1d7"
_RECEIPT_GIT_BLOB = "cd69e5d15c3ed122c0082ae5d7f87bc84c2311f7"
_RUN_ID = "contextual-expert-aggregation-frozen-policy-2024-audit-v2"
_RUNS = _CONTROL_ROOT / "runs"
_PRIVATE = _RUNS / f".sealing-{_RUN_ID}"
_FAILURE = _RUNS / f".failed-{_RUN_ID}"
_FINAL = _RUNS / _RUN_ID
_ATTEMPT_LOCK = _CONTROL_ROOT / "AUDIT_2024_ATTEMPT_LOCK.json"
_PENDING_MARKER = _CONTROL_ROOT / "AUDIT_2024_STAGE_SUCCESS.pending"
_SUCCESS_MARKER = _CONTROL_ROOT / "AUDIT_2024_STAGE_SUCCESS.json"

# This is intentionally literal.  It must equal the expanded inventory in the
# post-import control module; neither module discovers dependencies by scanning.
FROZEN_DEPENDENCY_PATHS = (
    Path(".gitattributes"),
    Path(".gitignore"),
    Path("requirements.txt"),
    Path("docs/aapl_causal_contextual_expert_aggregation_v2.md"),
    Path("docs/aapl_causal_contextual_expert_aggregation_v1.md"),
    Path("e/aapl_causal_contextual_expert_aggregation_v1/PREFLIGHT_REJECTED.md"),
    Path("docs/aapl_chronological_exhaustion_expert_v1.md"),
    Path("agent_benchmark/__init__.py"),
    Path("agent_benchmark/contextual_expert_aggregation_bootstrap.py"),
    Path("agent_benchmark/contextual_expert_aggregation.py"),
    Path("agent_benchmark/chronological_exhaustion_expert.py"),
    Path("agent_benchmark/contextual_expert_aggregation_replay.py"),
    Path("agent_benchmark/contextual_expert_aggregation_ledger.py"),
    Path("agent_benchmark/contextual_expert_aggregation_evaluation.py"),
    Path("agent_benchmark/contextual_expert_aggregation_artifacts.py"),
    Path("agent_benchmark/contextual_expert_aggregation_stage.py"),
    Path("agent_benchmark/contextual_expert_aggregation_verifier.py"),
    Path("agent_benchmark/contextual_expert_aggregation_experiment.py"),
    Path("tests/conftest.py"),
    Path("tests/test_contextual_expert_aggregation_bootstrap.py"),
    Path("tests/test_contextual_expert_aggregation.py"),
    Path("tests/test_chronological_exhaustion_expert.py"),
    Path("tests/test_contextual_expert_aggregation_replay.py"),
    Path("tests/test_contextual_expert_aggregation_ledger.py"),
    Path("tests/test_contextual_expert_aggregation_evaluation.py"),
    Path("tests/test_contextual_expert_aggregation_artifacts.py"),
    Path("tests/test_contextual_expert_aggregation_stage.py"),
    Path("tests/test_contextual_expert_aggregation_verifier.py"),
    Path("tests/test_contextual_expert_aggregation_experiment.py"),
    Path("README.md"),
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

_CRLF_EQUIVALENT_SUFFIXES = frozenset({".md", ".py", ".pyw"})
_CRLF_EQUIVALENT_CONTROLS = frozenset(
    {".gitattributes", ".gitignore", "requirements.txt"}
)
_BLOCKED_ROOT_IMPORTS = frozenset(
    {
        ".agents",
        ".codex",
        ".git",
        ".pytest_cache",
        ".venv",
        "ENV",
        "build",
        "data",
        "dist",
        "docs",
        "e",
        "env",
        "frontend",
        "node_modules",
        "prompts",
        "tests",
        "ui",
        "venv",
    }
)
_DISPATCH_TARGETS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "stage": (
            "agent_benchmark.contextual_expert_aggregation_2024_audit_runner",
            "run_stage",
        ),
        "verify": (
            "agent_benchmark.contextual_expert_aggregation_2024_audit_verifier",
            "verify_stage",
        ),
    }
)
_OBJECT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")

DispatchCallable = Callable[[str, str], tuple[Mapping[str, Any], int]]
ClockCallable = Callable[[], float]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise BootstrapSecurityError(
            "noncanonical_attestation",
            "bootstrap evidence is not finite canonical JSON",
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class BootstrapSecurityError(RuntimeError):
    """Fail-closed bootstrap error with immutable structured evidence."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details: Mapping[str, Any] = _freeze(details or {})

    def as_mapping(self) -> Mapping[str, Any]:
        return _freeze(
            {
                "bootstrap_id": BOOTSTRAP_ID,
                "code": self.code,
                "message": str(self),
                "details": self.details,
            }
        )


def _fail(code: str, message: str, **details: Any) -> None:
    raise BootstrapSecurityError(code, message, details=details)


def _parse_fixed_argv(argv: Sequence[str]) -> tuple[str, str]:
    values = tuple(argv)
    if len(values) != 2:
        _fail(
            "unsafe_argv",
            "exactly two positional arguments are required",
            expected=("{stage|verify}", "audit_2024"),
            received=values,
        )
    operation, stage = values
    if operation not in ALLOWED_OPERATIONS or stage != "audit_2024":
        _fail(
            "unsafe_argv",
            "only stage audit_2024 and verify audit_2024 are allowed",
            allowed_operations=ALLOWED_OPERATIONS,
            allowed_stages=ALLOWED_STAGES,
            received=values,
        )
    return operation, stage


def _is_reparse(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400))
    return bool(attributes & marker)


def _lstat(path: Path, *, label: str) -> os.stat_result:
    try:
        return os.lstat(path)
    except OSError as exc:
        _fail(
            "path_hardening_failed",
            f"{label} is missing or unreadable",
            path=str(path),
            error=type(exc).__name__,
        )


def _ordinary(path: Path, *, directory: bool, label: str) -> None:
    value = _lstat(path, label=label)
    correct_type = (
        stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    )
    try:
        mount = path.is_mount()
    except OSError:
        mount = True
    if (
        not correct_type
        or stat.S_ISLNK(value.st_mode)
        or _is_reparse(value)
        or mount
    ):
        _fail(
            "path_hardening_failed",
            f"{label} is not an ordinary non-reparse path",
            path=str(path),
        )


def _lexical_absolute(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        _fail(
            "path_hardening_failed",
            f"{label} contains a lexical parent traversal",
        )
    return Path(os.path.abspath(os.fspath(raw)))


def _safe_relative(path: Path, *, label: str) -> Path:
    value = Path(path)
    text = value.as_posix()
    if (
        value.is_absolute()
        or not value.parts
        or ".." in value.parts
        or ":" in text
        or "\\" in text
        or any(character in text for character in "\r\n\t\0")
    ):
        _fail("unsafe_literal_path", f"{label} is not a safe literal path")
    return value


def _descendant(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError:
        _fail(
            "path_hardening_failed",
            f"{label} escapes its frozen parent",
            path=str(path),
        )


def _harden_relative(
    root: Path,
    relative: Path,
    *,
    final_directory: bool,
    required: bool,
    label: str,
) -> Path:
    safe = _safe_relative(relative, label=label)
    target = _lexical_absolute(root / safe, label=label)
    _descendant(target, root, label=label)
    current = root
    for position, part in enumerate(safe.parts):
        current = current / part
        final = position == len(safe.parts) - 1
        if not os.path.lexists(current):
            if required:
                _fail(
                    "path_hardening_failed",
                    f"{label} is required but missing",
                    path=str(current),
                )
            return target
        _ordinary(
            current,
            directory=(not final or final_directory),
            label=label if final else f"{label} parent",
        )
    return target


def _harden_runtime_paths(repo_root: Path) -> Mapping[str, str]:
    root = _lexical_absolute(repo_root, label="repository root")
    _ordinary(root, directory=True, label="repository root")
    package = _harden_relative(
        root,
        Path("agent_benchmark"),
        final_directory=True,
        required=True,
        label="agent package",
    )
    bootstrap = _harden_relative(
        root,
        _BOOTSTRAP_RELATIVE_PATH,
        final_directory=False,
        required=True,
        label="bootstrap source",
    )
    control = _harden_relative(
        root,
        _CONTROL_ROOT,
        final_directory=True,
        required=True,
        label="control root",
    )
    authorized = _harden_relative(
        root,
        _AUTHORIZED_INPUTS,
        final_directory=True,
        required=True,
        label="authorized input directory",
    )
    input_path = _harden_relative(
        root,
        _INPUT_RELATIVE_PATH,
        final_directory=False,
        required=True,
        label="bounded input",
    )
    receipt = _harden_relative(
        root,
        _RECEIPT_RELATIVE_PATH,
        final_directory=False,
        required=True,
        label="sanitized receipt",
    )
    optional_specs = (
        ("attempt_lock", _ATTEMPT_LOCK, False),
        ("pending_marker", _PENDING_MARKER, False),
        ("success_marker", _SUCCESS_MARKER, False),
        ("runs", _RUNS, True),
        ("private", _PRIVATE, True),
        ("failure", _FAILURE, True),
        ("final", _FINAL, True),
    )
    result = {
        "repo_root": str(root),
        "agent_package": str(package),
        "bootstrap": str(bootstrap),
        "control_root": str(control),
        "authorized_inputs": str(authorized),
        "input": str(input_path),
        "receipt": str(receipt),
    }
    for name, relative, directory in optional_specs:
        result[name] = str(
            _harden_relative(
                root,
                relative,
                final_directory=directory,
                required=False,
                label=name.replace("_", " "),
            )
        )
    return _freeze(result)


def _repo_root_from_file() -> Path:
    source = _lexical_absolute(Path(__file__), label="bootstrap source")
    _ordinary(source, directory=False, label="bootstrap source")
    root = _lexical_absolute(source.parent.parent, label="repository root")
    _ordinary(root, directory=True, label="repository root")
    expected = root / _BOOTSTRAP_RELATIVE_PATH
    if source != expected:
        _fail(
            "foreign_bootstrap_location",
            "bootstrap source is not at its frozen repository-relative path",
            actual=str(source),
            expected=str(expected),
        )
    return root


def _validate_runtime(
    *,
    repo_root: Path,
    cwd: Path,
    isolated: int,
    dont_write_bytecode_flag: int,
    dont_write_bytecode_value: bool,
    optimize: int,
    orig_argv: Sequence[str],
    warn_options: Sequence[str],
    xoptions: Mapping[str, Any],
    operation: str,
    stage: str,
    sys_path: Sequence[str],
    preloaded_agent_modules: Sequence[str],
) -> Mapping[str, Any]:
    if int(isolated) != 1:
        _fail("isolated_mode_required", "Python isolated mode requires -I")
    if int(dont_write_bytecode_flag) != 1 or not dont_write_bytecode_value:
        _fail(
            "bytecode_writes_disabled_required",
            "Python bytecode writes must be disabled with -B",
        )
    expected_tail = (
        "-I",
        "-B",
        _BOOTSTRAP_RELATIVE_PATH.as_posix(),
        operation,
        stage,
    )
    observed = tuple(orig_argv)
    if (
        int(optimize) != 0
        or len(observed) < 2
        or observed[1:] != expected_tail
        or tuple(warn_options)
        or dict(xoptions)
    ):
        _fail(
            "noncanonical_interpreter_flags",
            "interpreter flags must exactly match the frozen -I -B command",
            expected_tail=expected_tail,
            observed_argv=observed,
        )
    resolved_cwd = _lexical_absolute(cwd, label="current working directory")
    _ordinary(resolved_cwd, directory=True, label="current working directory")
    if resolved_cwd != repo_root:
        _fail(
            "wrong_working_directory",
            "current working directory must equal the repository root",
            expected=str(repo_root),
            actual=str(resolved_cwd),
        )
    relative_entries: list[str] = []
    repository_entries: list[str] = []
    for raw in sys_path:
        if not raw:
            relative_entries.append(raw)
            continue
        entry = Path(raw)
        if not entry.is_absolute():
            relative_entries.append(raw)
            continue
        resolved = _lexical_absolute(entry, label="preimport sys.path entry")
        if resolved == repo_root:
            repository_entries.append(raw)
        else:
            try:
                resolved.relative_to(repo_root)
            except ValueError:
                pass
            else:
                repository_entries.append(raw)
    if relative_entries or repository_entries:
        _fail(
            "unsafe_preimport_sys_path",
            "relative or repository paths exist before attestation",
            relative_entries=tuple(relative_entries),
            repository_entries=tuple(repository_entries),
        )
    preloaded = tuple(sorted(preloaded_agent_modules))
    if preloaded:
        _fail(
            "package_imported_before_attestation",
            "agent_benchmark was imported before attestation",
            modules=preloaded,
        )
    return _freeze(
        {
            "isolated": True,
            "dont_write_bytecode": True,
            "optimize": 0,
            "orig_argv_tail": expected_tail,
            "warn_options": (),
            "xoptions": {},
            "cwd_equals_repository_root": True,
            "preimport_sys_path_clean": True,
            "agent_package_preloaded": False,
            "dispatch_environment_sanitized": True,
            "dispatch_sys_path_sanitized": True,
        }
    )


def _find_git(repo_root: Path) -> Path:
    found = shutil.which("git")
    if not found:
        _fail("git_unavailable", "Git executable was not found")
    try:
        executable = Path(found).resolve(strict=True)
    except OSError as exc:
        _fail("git_unavailable", "Git executable could not be resolved")
        raise AssertionError from exc
    try:
        executable.relative_to(repo_root)
    except ValueError:
        return executable
    _fail(
        "foreign_git_executable",
        "Git executable resolves inside the repository",
        executable=str(executable),
    )


def _sanitized_environment() -> dict[str, str]:
    result = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("GIT_", "PYTHON"))
    }
    result.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return result


def _run_git(
    executable: Path,
    repo_root: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        value = subprocess.run(
            [
                str(executable),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(repo_root),
                *args,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_sanitized_environment(),
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(
            "git_inspection_failed",
            "exact-path Git inspection could not run",
            command=tuple(args[:2]),
            error=type(exc).__name__,
        )
    if check and value.returncode != 0:
        _fail(
            "git_inspection_failed",
            "exact-path Git inspection failed",
            command=tuple(args[:2]),
            returncode=value.returncode,
        )
    return value


def _git_text(executable: Path, repo_root: Path, args: Sequence[str]) -> str:
    try:
        return _run_git(executable, repo_root, args).stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        _fail("git_parse_failed", "Git metadata is not strict UTF-8")
        raise AssertionError from exc


def _literal_pathspec(relative: str) -> str:
    safe = _safe_relative(Path(relative), label="Git path")
    return f":(top,literal){safe.as_posix()}"


def _index_entry(
    executable: Path, repo_root: Path, relative: str
) -> tuple[str, str]:
    raw = _run_git(
        executable,
        repo_root,
        ("ls-files", "--stage", "-z", "--", _literal_pathspec(relative)),
    ).stdout
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1:
        _fail(
            "dependency_identity_changed",
            "dependency lacks one exact stage-zero index entry",
            path=relative,
        )
    metadata, separator, encoded_path = records[0].partition(b"\t")
    try:
        mode, object_id, stage = metadata.decode("ascii").split()
        observed_path = encoded_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        _fail("git_parse_failed", "dependency index entry is malformed")
        raise AssertionError from exc
    if (
        not separator
        or stage != "0"
        or observed_path != relative
        or mode not in {"100644", "100755"}
        or _OBJECT_RE.fullmatch(object_id) is None
    ):
        _fail(
            "dependency_identity_changed",
            "dependency index entry changed",
            path=relative,
        )
    return mode, object_id


def _head_entry(
    executable: Path, repo_root: Path, relative: str
) -> tuple[str, str]:
    raw = _run_git(
        executable,
        repo_root,
        ("ls-tree", "-z", "HEAD", "--", _literal_pathspec(relative)),
    ).stdout
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1:
        _fail(
            "dependency_identity_changed",
            "dependency lacks one exact HEAD tree entry",
            path=relative,
        )
    metadata, separator, encoded_path = records[0].partition(b"\t")
    try:
        mode, object_type, object_id = metadata.decode("ascii").split()
        observed_path = encoded_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        _fail("git_parse_failed", "dependency HEAD entry is malformed")
        raise AssertionError from exc
    if (
        not separator
        or observed_path != relative
        or object_type != "blob"
        or mode not in {"100644", "100755"}
        or _OBJECT_RE.fullmatch(object_id) is None
    ):
        _fail(
            "dependency_identity_changed",
            "dependency HEAD entry changed",
            path=relative,
        )
    return mode, object_id


def _attest_dependency(
    executable: Path, repo_root: Path, relative_path: Path
) -> Mapping[str, str]:
    relative = _safe_relative(relative_path, label="frozen dependency").as_posix()
    path = _harden_relative(
        repo_root,
        relative_path,
        final_directory=False,
        required=True,
        label="frozen dependency",
    )
    index_mode, index_blob = _index_entry(executable, repo_root, relative)
    head_mode, head_blob = _head_entry(executable, repo_root, relative)
    flags = _run_git(
        executable,
        repo_root,
        ("ls-files", "-v", "-z", "--", _literal_pathspec(relative)),
    ).stdout
    expected_flag = f"H {relative}".encode("utf-8")
    if flags.rstrip(b"\0") != expected_flag:
        _fail(
            "dependency_identity_changed",
            "dependency index flags changed",
            path=relative,
        )
    try:
        local = path.read_bytes()
    except OSError as exc:
        _fail(
            "dependency_identity_changed",
            "dependency worktree bytes are unreadable",
            path=relative,
        )
        raise AssertionError from exc
    head = _run_git(executable, repo_root, ("show", f"HEAD:{relative}")).stdout
    indexed = _run_git(executable, repo_root, ("show", f":{relative}")).stdout
    normalized = (
        local.replace(b"\r\n", b"\n")
        if relative in _CRLF_EQUIVALENT_CONTROLS
        or relative_path.suffix.lower() in _CRLF_EQUIVALENT_SUFFIXES
        else local
    )
    if (
        head_mode != index_mode
        or head_blob != index_blob
        or head != indexed
        or head != normalized
    ):
        _fail(
            "dependency_identity_changed",
            "dependency differs across HEAD, index, and worktree",
            path=relative,
        )
    return _freeze(
        {
            "head_blob": head_blob,
            "index_blob": index_blob,
            "worktree_raw_sha256": _sha256_bytes(local),
        }
    )


def _unopened_git_identity(
    executable: Path,
    repo_root: Path,
    relative_path: Path,
    *,
    expected_blob: str,
) -> Mapping[str, str]:
    relative = _safe_relative(relative_path, label="unopened control").as_posix()
    _, index_blob = _index_entry(executable, repo_root, relative)
    head_blob = _git_text(
        executable, repo_root, ("rev-parse", f"HEAD:{relative}")
    )
    if head_blob != expected_blob or index_blob != expected_blob:
        _fail(
            "unopened_control_identity_changed",
            "unopened input or receipt HEAD/index object changed",
            path=relative,
        )
    return _freeze(
        {"path": relative, "head_blob": head_blob, "index_blob": index_blob}
    )


def _attest_git_and_dependencies(repo_root: Path) -> Mapping[str, Any]:
    dependencies = tuple(FROZEN_DEPENDENCY_PATHS)
    if (
        not dependencies
        or len(dependencies) != len(set(dependencies))
        or len({path.as_posix().casefold() for path in dependencies})
        != len(dependencies)
        or _INPUT_RELATIVE_PATH in dependencies
        or _RECEIPT_RELATIVE_PATH in dependencies
    ):
        _fail(
            "invalid_dependency_allowlist",
            "frozen dependency inventory is empty, duplicate, or unsafe",
        )
    git = _find_git(repo_root)
    top = _lexical_absolute(
        Path(_git_text(git, repo_root, ("rev-parse", "--show-toplevel"))),
        label="Git top level",
    )
    branch = _git_text(git, repo_root, ("symbolic-ref", "--quiet", "--short", "HEAD"))
    commit = _git_text(git, repo_root, ("rev-parse", "HEAD"))
    upstream = _git_text(
        git,
        repo_root,
        ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"),
    )
    upstream_commit = _git_text(git, repo_root, ("rev-parse", "@{upstream}"))
    origin = _git_text(git, repo_root, ("remote", "get-url", "origin"))
    ancestry = _run_git(
        git,
        repo_root,
        (
            "merge-base",
            "--is-ancestor",
            _PREREGISTRATION_COMMIT,
            commit,
        ),
        check=False,
    )
    if (
        top != repo_root
        or branch != _EXPECTED_BRANCH
        or _OBJECT_RE.fullmatch(commit) is None
        or upstream != f"origin/{branch}"
        or upstream_commit != commit
        or origin not in _EXPECTED_ORIGIN_URLS
        or ancestry.returncode != 0
    ):
        _fail(
            "git_identity_changed",
            "bootstrap requires the exact pushed audit identity and ancestry",
        )
    files = {
        path.as_posix(): _attest_dependency(git, repo_root, path)
        for path in dependencies
    }
    dependency_identity = {
        "schema_version": 1,
        "files": {key: files[key] for key in sorted(files)},
    }
    return _freeze(
        {
            "branch": branch,
            "commit": commit,
            "upstream": upstream,
            "upstream_commit": upstream_commit,
            "origin_url": origin,
            "origin_repository": _EXPECTED_ORIGIN_REPOSITORY,
            "preregistration_commit": _PREREGISTRATION_COMMIT,
            "preregistration_is_ancestor": True,
            "head_equals_upstream": True,
            "dependency_identity": dependency_identity,
            "dependency_identity_sha256": _sha256_bytes(
                _canonical_bytes(dependency_identity)
            ),
            "input_git_identity": _unopened_git_identity(
                git,
                repo_root,
                _INPUT_RELATIVE_PATH,
                expected_blob=_INPUT_GIT_BLOB,
            ),
            "receipt_git_identity": _unopened_git_identity(
                git,
                repo_root,
                _RECEIPT_RELATIVE_PATH,
                expected_blob=_RECEIPT_GIT_BLOB,
            ),
            "prelock_input_worktree_bytes_opened": False,
            "prelock_receipt_worktree_bytes_opened": False,
        }
    )


def _attest_preimport_environment(
    *,
    repo_root: Path,
    hardened_paths: Mapping[str, str],
    cwd: Path,
    isolated: int,
    dont_write_bytecode_flag: int,
    dont_write_bytecode_value: bool,
    optimize: int,
    orig_argv: Sequence[str],
    warn_options: Sequence[str],
    xoptions: Mapping[str, Any],
    sys_path: Sequence[str],
    preloaded_agent_modules: Sequence[str],
    operation: str,
    stage: str,
    monotonic_start: float,
) -> Mapping[str, Any]:
    if type(monotonic_start) is not float or not math.isfinite(monotonic_start):
        _fail(
            "invalid_bootstrap_clock",
            "bootstrap monotonic start must be a finite float",
        )
    root = _lexical_absolute(repo_root, label="repository root")
    runtime = _validate_runtime(
        repo_root=root,
        cwd=cwd,
        isolated=isolated,
        dont_write_bytecode_flag=dont_write_bytecode_flag,
        dont_write_bytecode_value=dont_write_bytecode_value,
        optimize=optimize,
        orig_argv=orig_argv,
        warn_options=warn_options,
        xoptions=xoptions,
        operation=operation,
        stage=stage,
        sys_path=sys_path,
        preloaded_agent_modules=preloaded_agent_modules,
    )
    git_identity = _attest_git_and_dependencies(root)
    body: dict[str, Any] = {
        "attestation_schema_version": ATTESTATION_SCHEMA_VERSION,
        "bootstrap_id": BOOTSTRAP_ID,
        "invocation": {"operation": operation, "stage": stage},
        "monotonic_start": monotonic_start,
        "monotonic_clock": "time.monotonic",
        "strict_deadline_seconds": STAGE_DEADLINE_SECONDS,
        "runtime_paths": dict(hardened_paths),
        "runtime": runtime,
        "git_identity": git_identity,
        "execution_surface": {
            "literal_dependency_paths": tuple(
                path.as_posix() for path in FROZEN_DEPENDENCY_PATHS
            ),
            "blocked_root_imports": tuple(sorted(_BLOCKED_ROOT_IMPORTS)),
            "unlisted_agent_modules_blocked": True,
            "control_tree_blocked_as_import_surface": True,
            "broad_repository_enumeration_performed": False,
        },
        "checks": {
            "fixed_argv": True,
            "all_runtime_paths_hardened_before_package_import": True,
            "runtime_isolated_before_import": True,
            "cwd_is_repo_root": True,
            "literal_dependencies_head_index_worktree_equal": True,
            "bounded_input_worktree_unopened": True,
            "receipt_worktree_unopened": True,
            "exact_pathspec_git_only": True,
            "dispatch_environment_sanitized": True,
            "dispatch_sys_path_sanitized": True,
            "control_tree_imports_blocked": True,
        },
    }
    body["attestation_sha256"] = _sha256_bytes(_canonical_bytes(body))
    return _freeze(body)


def _allowed_agent_modules() -> frozenset[str]:
    result = {_AGENT_PACKAGE}
    prefix = f"{_AGENT_PACKAGE}/"
    for path in FROZEN_DEPENDENCY_PATHS:
        relative = path.as_posix()
        if not relative.startswith(prefix) or path.suffix.lower() != ".py":
            continue
        module_parts = list(path.with_suffix("").parts)
        if module_parts[-1] == "__init__":
            module_parts.pop()
        result.add(".".join(module_parts))
    return frozenset(result)


def _repository_top_level_candidate(repo_root: Path, name: str) -> bool:
    if not name.isidentifier():
        return False
    candidates = [
        repo_root / f"{name}.py",
        repo_root / f"{name}.pyw",
        repo_root / f"{name}.pyc",
        repo_root / name,
        repo_root / name / "__init__.py",
    ]
    candidates.extend(
        repo_root / f"{name}{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    return any(os.path.lexists(path) for path in candidates)


class _BlockedRepositoryImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root
        self._allowed_agent = _allowed_agent_modules()

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any = None,
    ) -> None:
        del path, target
        top = fullname.split(".", 1)[0]
        if top in _BLOCKED_ROOT_IMPORTS:
            _fail(
                "blocked_control_tree_import",
                "the repository control/data tree is not an import surface",
                fullname=fullname,
            )
        if (
            top == _AGENT_PACKAGE
            and fullname not in self._allowed_agent
        ):
            _fail(
                "unattested_agent_import",
                "an unlisted agent module import was blocked",
                fullname=fullname,
            )
        if (
            "." not in fullname
            and top != _AGENT_PACKAGE
            and _repository_top_level_candidate(self._root, top)
        ):
            _fail(
                "unattested_repository_import",
                "an unlisted repository-root module import was blocked",
                fullname=fullname,
            )
        return None


def require_active_attestation(
    operation: str, stage: str, repo_root: Path
) -> Mapping[str, Any]:
    """Return the exact process-local bootstrap proof during guarded dispatch."""

    value = getattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE, None)
    if not isinstance(value, Mapping):
        _fail(
            "missing_active_attestation",
            "audit runner and verifier require the isolated 2024 bootstrap",
        )
    thawed = _thaw(value)
    supplied_hash = thawed.pop("attestation_sha256", None)
    expected_root = str(
        _lexical_absolute(repo_root, label="active attestation repository root")
    )
    git_identity = thawed.get("git_identity")
    runtime = thawed.get("runtime")
    checks = thawed.get("checks")
    execution_surface = thawed.get("execution_surface")
    input_identity = (
        git_identity.get("input_git_identity")
        if isinstance(git_identity, dict)
        else None
    )
    receipt_identity = (
        git_identity.get("receipt_git_identity")
        if isinstance(git_identity, dict)
        else None
    )
    dependency_identity = (
        git_identity.get("dependency_identity")
        if isinstance(git_identity, dict)
        else None
    )
    if (
        operation not in ALLOWED_OPERATIONS
        or stage != "audit_2024"
        or supplied_hash != _sha256_bytes(_canonical_bytes(thawed))
        or thawed.get("attestation_schema_version")
        != ATTESTATION_SCHEMA_VERSION
        or thawed.get("bootstrap_id") != BOOTSTRAP_ID
        or thawed.get("invocation")
        != {"operation": operation, "stage": stage}
        or thawed.get("monotonic_clock") != "time.monotonic"
        or type(thawed.get("monotonic_start")) is not float
        or not math.isfinite(thawed["monotonic_start"])
        or thawed.get("strict_deadline_seconds") != STAGE_DEADLINE_SECONDS
        or not isinstance(thawed.get("runtime_paths"), dict)
        or thawed["runtime_paths"].get("repo_root") != expected_root
        or not isinstance(git_identity, dict)
        or git_identity.get("branch") != _EXPECTED_BRANCH
        or git_identity.get("upstream") != f"origin/{_EXPECTED_BRANCH}"
        or git_identity.get("commit") != git_identity.get("upstream_commit")
        or _OBJECT_RE.fullmatch(str(git_identity.get("commit"))) is None
        or git_identity.get("origin_url") not in _EXPECTED_ORIGIN_URLS
        or git_identity.get("preregistration_commit")
        != _PREREGISTRATION_COMMIT
        or git_identity.get("preregistration_is_ancestor") is not True
        or git_identity.get("head_equals_upstream") is not True
        or git_identity.get("prelock_input_worktree_bytes_opened") is not False
        or git_identity.get("prelock_receipt_worktree_bytes_opened") is not False
        or _SHA256_RE.fullmatch(
            str(git_identity.get("dependency_identity_sha256"))
        )
        is None
        or not isinstance(dependency_identity, dict)
        or set(dependency_identity) != {"schema_version", "files"}
        or dependency_identity.get("schema_version") != 1
        or not isinstance(dependency_identity.get("files"), dict)
        or git_identity.get("dependency_identity_sha256")
        != _sha256_bytes(_canonical_bytes(dependency_identity))
        or input_identity
        != {
            "path": _INPUT_RELATIVE_PATH.as_posix(),
            "head_blob": _INPUT_GIT_BLOB,
            "index_blob": _INPUT_GIT_BLOB,
        }
        or receipt_identity
        != {
            "path": _RECEIPT_RELATIVE_PATH.as_posix(),
            "head_blob": _RECEIPT_GIT_BLOB,
            "index_blob": _RECEIPT_GIT_BLOB,
        }
        or not isinstance(runtime, dict)
        or runtime.get("isolated") is not True
        or runtime.get("dont_write_bytecode") is not True
        or runtime.get("cwd_equals_repository_root") is not True
        or runtime.get("preimport_sys_path_clean") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(execution_surface, dict)
        or execution_surface.get("blocked_root_imports")
        != sorted(_BLOCKED_ROOT_IMPORTS)
        or execution_surface.get("unlisted_agent_modules_blocked") is not True
        or execution_surface.get("control_tree_blocked_as_import_surface")
        is not True
        or execution_surface.get("broad_repository_enumeration_performed")
        is not False
        or tuple(
            execution_surface.get("literal_dependency_paths", ())
        )
        != tuple(path.as_posix() for path in FROZEN_DEPENDENCY_PATHS)
        or sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or not sys.dont_write_bytecode
    ):
        _fail(
            "invalid_active_attestation",
            "active bootstrap evidence does not match this invocation",
            operation=operation,
            stage=stage,
        )
    return value


def bootstrap_elapsed(
    attestation: Mapping[str, Any], *, clock: ClockCallable | None = None
) -> float:
    """Measure elapsed time from the immutable bootstrap-entry sample."""

    if not isinstance(attestation, Mapping):
        _fail("invalid_bootstrap_clock", "bootstrap attestation is not a mapping")
    start = attestation.get("monotonic_start")
    if type(start) is not float or not math.isfinite(start):
        _fail("invalid_bootstrap_clock", "bootstrap start is not a finite float")
    selected = time.monotonic if clock is None else clock
    current = selected()
    if type(current) is not float or not math.isfinite(current) or current < start:
        _fail("invalid_bootstrap_clock", "bootstrap monotonic clock moved backwards")
    return current - start


def require_bootstrap_deadline(
    attestation: Mapping[str, Any],
    phase: str = "bootstrap_exit",
    *,
    clock: ClockCallable | None = None,
) -> float:
    """Require one named sample to remain strictly below 1,800 seconds."""

    if phase not in BOOTSTRAP_DEADLINE_PHASES:
        _fail("invalid_deadline_phase", "bootstrap deadline phase is not frozen")
    elapsed = bootstrap_elapsed(attestation, clock=clock)
    if not elapsed < STAGE_DEADLINE_SECONDS:
        _fail(
            "bootstrap_deadline_exceeded",
            "bootstrap elapsed time is not strictly below 1800 seconds",
            phase=phase,
            elapsed_seconds=elapsed,
            strict_limit_seconds=STAGE_DEADLINE_SECONDS,
        )
    return elapsed


def _import_dispatch(operation: str, stage: str) -> tuple[Mapping[str, Any], int]:
    module_name, callable_name = _DISPATCH_TARGETS[operation]
    module = importlib.import_module(module_name)
    target = getattr(module, callable_name, None)
    if not callable(target):
        _fail(
            "dispatch_target_missing",
            "the fixed 2024 audit dispatch target is unavailable",
            module=module_name,
            callable=callable_name,
        )
    result = target(stage)
    if not isinstance(result, Mapping):
        _fail(
            "invalid_dispatch_result",
            "the fixed audit target did not return a mapping",
            result_type=type(result).__name__,
        )
    exit_code = 0
    if operation == "stage" and not bool(result.get("stage_pass")):
        exit_code = 2
    return result, exit_code


def _safe_dispatch_sys_path(repo_root: Path) -> list[str]:
    safe: list[str] = []
    for raw in sys.path:
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            continue
        resolved = _lexical_absolute(path, label="dispatch sys.path entry")
        try:
            resolved.relative_to(repo_root)
        except ValueError:
            safe.append(raw)
    return [str(repo_root), *safe]


def _guarded_main(
    argv: Sequence[str],
    *,
    repo_root: Path,
    cwd: Path,
    isolated: int,
    dont_write_bytecode_flag: int,
    dont_write_bytecode_value: bool,
    optimize: int,
    orig_argv: Sequence[str],
    warn_options: Sequence[str],
    xoptions: Mapping[str, Any],
    preimport_sys_path: Sequence[str],
    preloaded_agent_modules: Sequence[str],
    dispatch: DispatchCallable,
    clock: ClockCallable = time.monotonic,
    monotonic_start: float | None = None,
) -> tuple[Mapping[str, Any], int]:
    start = clock() if monotonic_start is None else monotonic_start
    operation, stage = _parse_fixed_argv(argv)
    root = _lexical_absolute(repo_root, label="repository root")
    hardened_paths = _harden_runtime_paths(root)
    attestation = _attest_preimport_environment(
        repo_root=root,
        hardened_paths=hardened_paths,
        cwd=cwd,
        isolated=isolated,
        dont_write_bytecode_flag=dont_write_bytecode_flag,
        dont_write_bytecode_value=dont_write_bytecode_value,
        optimize=optimize,
        orig_argv=orig_argv,
        warn_options=warn_options,
        xoptions=xoptions,
        sys_path=preimport_sys_path,
        preloaded_agent_modules=preloaded_agent_modules,
        operation=operation,
        stage=stage,
        monotonic_start=start,
    )
    if hasattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE):
        _fail(
            "nested_active_attestation",
            "a 2024 audit bootstrap attestation is already active",
        )
    setattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE, attestation)
    original_environment = dict(os.environ)
    original_sys_path = list(sys.path)
    original_meta_path = list(sys.meta_path)
    blocker = _BlockedRepositoryImportFinder(root)
    bootstrap_exit: Mapping[str, Any] | None = None
    sanitized_environment = _sanitized_environment()
    os.environ.clear()
    os.environ.update(sanitized_environment)
    sys.path[:] = _safe_dispatch_sys_path(root)
    sys.meta_path.insert(0, blocker)
    try:
        result, exit_code = dispatch(operation, stage)
        if operation == "verify":
            elapsed = require_bootstrap_deadline(
                attestation, "bootstrap_exit", clock=clock
            )
            bootstrap_exit = _freeze(
                {
                    "phase": "bootstrap_exit",
                    "elapsed_seconds": elapsed,
                    "strict_limit_seconds": STAGE_DEADLINE_SECONDS,
                    "passed": True,
                }
            )
    finally:
        sys.path[:] = original_sys_path
        sys.meta_path[:] = original_meta_path
        os.environ.clear()
        os.environ.update(original_environment)
        try:
            delattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE)
        except AttributeError:
            pass
    if not isinstance(result, Mapping):
        _fail(
            "invalid_dispatch_result",
            "guarded audit dispatch did not return a mapping",
            result_type=type(result).__name__,
        )
    if (
        not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or not 0 <= exit_code <= 255
    ):
        _fail(
            "invalid_dispatch_exit_code",
            "guarded audit dispatch returned an invalid exit code",
            exit_code=repr(exit_code),
        )
    return (
        _freeze(
            {
                "bootstrap_attestation": attestation,
                "bootstrap_exit": bootstrap_exit,
                "result": result,
            }
        ),
        exit_code,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Harden and attest the process, then invoke one fixed audit callable."""

    clock = time.monotonic
    monotonic_start = clock()
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    root = _repo_root_from_file()
    # Do this immediately after deriving/hardening the invoked source location.
    # `_guarded_main` repeats it so direct programmatic calls have the same
    # contract, but production never touches cwd or a package import first.
    _harden_runtime_paths(root)
    payload, exit_code = _guarded_main(
        arguments,
        repo_root=root,
        cwd=Path.cwd(),
        isolated=sys.flags.isolated,
        dont_write_bytecode_flag=sys.flags.dont_write_bytecode,
        dont_write_bytecode_value=sys.dont_write_bytecode,
        optimize=sys.flags.optimize,
        orig_argv=tuple(sys.orig_argv),
        warn_options=tuple(sys.warnoptions),
        xoptions=dict(sys._xoptions),
        preimport_sys_path=tuple(sys.path),
        preloaded_agent_modules=tuple(
            name
            for name in sys.modules
            if name == _AGENT_PACKAGE or name.startswith(f"{_AGENT_PACKAGE}.")
        ),
        dispatch=_import_dispatch,
        clock=clock,
        monotonic_start=monotonic_start,
    )
    print(json.dumps(_thaw(payload), indent=2, sort_keys=True, allow_nan=False))
    return exit_code


def _cli() -> int:
    try:
        return main()
    except BootstrapSecurityError as exc:
        print(
            json.dumps(
                {"bootstrap_error": _thaw(exc.as_mapping())},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 64


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "ALLOWED_OPERATIONS",
    "ALLOWED_STAGES",
    "ATTESTATION_SCHEMA_VERSION",
    "BOOTSTRAP_DEADLINE_PHASES",
    "BOOTSTRAP_ID",
    "BootstrapSecurityError",
    "ClockCallable",
    "DispatchCallable",
    "FROZEN_DEPENDENCY_PATHS",
    "STAGE_DEADLINE_SECONDS",
    "bootstrap_elapsed",
    "main",
    "require_active_attestation",
    "require_bootstrap_deadline",
]
