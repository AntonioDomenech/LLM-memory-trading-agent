"""Stdlib-only isolated bootstrap for the one-shot continuation audit.

The only supported commands are exactly::

    python -I -B \
        agent_benchmark/contextual_expert_aggregation_audit_bootstrap.py \
        stage audit
    python -I -B \
        agent_benchmark/contextual_expert_aggregation_audit_bootstrap.py \
        verify audit

No project package is imported at module load time.  Before dispatch, the
bootstrap discovers every Python/native candidate that could be imported from
the repository execution surface and proves that each candidate is a regular,
tracked file with identical HEAD, index, and worktree content.  Only then is
the repository root temporarily added to a sanitized ``sys.path`` and the
fixed runner or verifier imported.

This module does not name, inspect, hash, or parse any market-data file.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


ATTESTATION_SCHEMA_VERSION = 1
BOOTSTRAP_ID = "contextual-expert-aggregation-audit-isolated-bootstrap-v2"
ALLOWED_OPERATIONS = ("stage", "verify")
ALLOWED_STAGES = ("audit",)
IMPORT_CANDIDATE_SUFFIXES = (
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".dylib",
    ".exe",
    ".py",
    ".pyc",
    ".pyd",
    ".pyo",
    ".pyw",
    ".so",
)

_AGENT_PACKAGE = "agent_benchmark"
_BOOTSTRAP_RELATIVE_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_audit_bootstrap.py"
)
_ACTIVE_ATTESTATION_ATTRIBUTE = (
    "_contextual_expert_aggregation_audit_bootstrap_attestation_v2"
)
_CONTROL_PATHS = (
    Path(".gitattributes"),
    Path(".gitignore"),
    Path("requirements.txt"),
)
_CONTROL_RELATIVES = tuple(path.as_posix() for path in _CONTROL_PATHS)
_CRLF_EQUIVALENT_SUFFIXES = frozenset({".py", ".pyw"})
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
            "agent_benchmark.contextual_expert_aggregation_audit_runner",
            "run_stage",
        ),
        "verify": (
            "agent_benchmark.contextual_expert_aggregation_audit_verifier",
            "verify_stage",
        ),
    }
)

DispatchCallable = Callable[[str, str], tuple[Mapping[str, Any], int]]


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


class _BlockedRootImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, blocked: Sequence[str]) -> None:
        self._blocked = frozenset(blocked)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any = None,
    ) -> None:
        top_level = fullname.split(".", 1)[0]
        if top_level in self._blocked:
            _fail(
                "blocked_root_import",
                "an unattested repository-root namespace import was blocked",
                fullname=fullname,
                blocked_root=top_level,
            )
        return None


def _parse_fixed_argv(argv: Sequence[str]) -> tuple[str, str]:
    values = tuple(argv)
    if len(values) != 2:
        _fail(
            "unsafe_argv",
            "exactly two positional arguments are required",
            expected=("{stage|verify}", "audit"),
            received=values,
        )
    operation, stage = values
    if operation not in ALLOWED_OPERATIONS or stage != "audit":
        _fail(
            "unsafe_argv",
            "only the fixed stage audit and verify audit commands are allowed",
            allowed_operations=ALLOWED_OPERATIONS,
            allowed_stages=ALLOWED_STAGES,
            received=values,
        )
    return operation, stage


def _is_reparse(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        _fail(
            "filesystem_inspection_failed",
            "an execution-surface path could not be inspected",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )


def _is_link_or_reparse(path: Path) -> bool:
    value = _lstat(path)
    return stat.S_ISLNK(value.st_mode) or _is_reparse(value)


def _resolved_directory(path: Path, *, label: str) -> Path:
    try:
        value = path.resolve(strict=True)
    except OSError as exc:
        _fail(
            "invalid_directory",
            f"{label} cannot be resolved",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )
    if not value.is_dir():
        _fail("invalid_directory", f"{label} is not a directory", path=str(value))
    return value


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _repo_root_from_file() -> Path:
    source = Path(os.path.abspath(__file__))
    if _is_link_or_reparse(source):
        _fail(
            "foreign_bootstrap_location",
            "bootstrap source is a symlink or reparse point",
            source=str(source),
        )
    resolved = source.resolve(strict=True)
    root = resolved.parent.parent
    expected = root / _BOOTSTRAP_RELATIVE_PATH
    if resolved != expected.resolve(strict=True):
        _fail(
            "foreign_bootstrap_location",
            "bootstrap source is not at its frozen repository-relative path",
            actual=str(resolved),
            expected=str(expected),
        )
    return _resolved_directory(root, label="repository root")


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
        _fail(
            "isolated_mode_required",
            "Python isolated mode is required; invoke with -I",
        )
    if int(dont_write_bytecode_flag) != 1 or not dont_write_bytecode_value:
        _fail(
            "bytecode_writes_disabled_required",
            "Python bytecode writes must be disabled; invoke with -B",
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
            optimize=int(optimize),
            warn_options=tuple(warn_options),
            xoptions=dict(xoptions),
        )
    resolved_cwd = _resolved_directory(cwd, label="current working directory")
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
        try:
            resolved = entry.resolve(strict=False)
        except OSError:
            resolved = entry.absolute()
        if resolved == repo_root or _path_is_within(resolved, repo_root):
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
        _fail("git_unavailable", "Git executable could not be resolved", error=str(exc))
    if executable == repo_root or _path_is_within(executable, repo_root):
        _fail(
            "foreign_git_executable",
            "Git executable resolves inside the repository",
            executable=str(executable),
        )
    return executable


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
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(
            "git_inspection_failed",
            "Git inspection could not run",
            args=tuple(args),
            error=f"{type(exc).__name__}: {exc}",
        )
    if check and value.returncode != 0:
        _fail(
            "git_inspection_failed",
            "Git inspection failed",
            args=tuple(args),
            returncode=value.returncode,
            stderr=value.stderr.decode("utf-8", errors="replace").strip(),
        )
    return value


def _git_text(executable: Path, repo_root: Path, args: Sequence[str]) -> str:
    try:
        return _run_git(executable, repo_root, args).stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        _fail("git_parse_failed", "Git metadata is not UTF-8", error=str(exc))


def _parse_index(raw: bytes) -> dict[str, tuple[str, str, int]]:
    result: dict[str, tuple[str, str, int]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, path_bytes = record.partition(b"\t")
        if not separator:
            _fail("git_parse_failed", "malformed index record")
        parts = metadata.decode("ascii").split()
        if len(parts) != 3:
            _fail("git_parse_failed", "malformed index metadata")
        mode, object_id, stage_text = parts
        relative = path_bytes.decode("utf-8", errors="surrogateescape")
        if relative in result:
            _fail(
                "unmerged_execution_surface",
                "multiple index stages exist for an execution candidate",
                path=relative,
            )
        result[relative] = (mode, object_id, int(stage_text))
    return result


def _parse_head(raw: bytes) -> dict[str, tuple[str, str, str]]:
    result: dict[str, tuple[str, str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, path_bytes = record.partition(b"\t")
        if not separator:
            _fail("git_parse_failed", "malformed HEAD tree record")
        parts = metadata.decode("ascii").split()
        if len(parts) != 3:
            _fail("git_parse_failed", "malformed HEAD tree metadata")
        mode, object_type, object_id = parts
        relative = path_bytes.decode("utf-8", errors="surrogateescape")
        result[relative] = (mode, object_type, object_id)
    return result


def _parse_flags(raw: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        text = record.decode("utf-8", errors="surrogateescape")
        marker, separator, relative = text.partition(" ")
        if not separator or len(marker) != 1:
            _fail("git_parse_failed", "malformed index flag record")
        result[relative] = marker
    return result


def _candidate_suffix(path: Path) -> bool:
    return path.suffix.lower() in IMPORT_CANDIDATE_SUFFIXES


def _scan_tree(directory: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    stack = [directory]
    while stack:
        current = stack.pop()
        if _is_link_or_reparse(current):
            _fail(
                "unsafe_execution_surface_link",
                "an execution-surface directory is a link or reparse point",
                path=str(current),
            )
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name.casefold())
        except OSError as exc:
            _fail(
                "filesystem_inspection_failed",
                "an execution-surface directory could not be enumerated",
                path=str(current),
                error=f"{type(exc).__name__}: {exc}",
            )
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                _fail(
                    "filesystem_inspection_failed",
                    "an execution-surface entry could not be inspected",
                    path=str(path),
                    error=f"{type(exc).__name__}: {exc}",
                )
            if entry.is_symlink() or _is_reparse(info):
                _fail(
                    "unsafe_execution_surface_link",
                    "an execution-surface entry is a link or reparse point",
                    path=str(path),
                )
            if stat.S_ISDIR(info.st_mode):
                stack.append(path)
            elif _candidate_suffix(path):
                if not stat.S_ISREG(info.st_mode):
                    _fail(
                        "unsafe_execution_surface_type",
                        "an execution candidate is not a regular file",
                        path=str(path),
                    )
                result.append(path)
    return tuple(result)


def _execution_surface(
    repo_root: Path,
    versioned_paths: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    candidates: set[Path] = {repo_root / path for path in _CONTROL_PATHS}
    package_roots: set[Path] = set()
    blocked: set[str] = set()
    agent_root = repo_root / _AGENT_PACKAGE
    if not agent_root.is_dir():
        _fail(
            "missing_agent_package",
            "the agent_benchmark package directory is missing",
            path=str(agent_root),
        )
    try:
        entries = sorted(os.scandir(repo_root), key=lambda item: item.name.casefold())
    except OSError as exc:
        _fail(
            "filesystem_inspection_failed",
            "repository root could not be enumerated",
            error=f"{type(exc).__name__}: {exc}",
        )
    for entry in entries:
        path = Path(entry.path)
        info = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or _is_reparse(info):
            _fail(
                "unsafe_execution_surface_link",
                "a repository-root entry is a link or reparse point",
                path=str(path),
            )
        if stat.S_ISREG(info.st_mode) and _candidate_suffix(path):
            candidates.add(path)
        elif stat.S_ISDIR(info.st_mode):
            if entry.name in _BLOCKED_ROOT_IMPORTS:
                blocked.add(entry.name)
            else:
                package_roots.add(path)
    package_roots.add(agent_root)
    for package_root in sorted(package_roots, key=lambda path: str(path).casefold()):
        if package_root.exists():
            candidates.update(_scan_tree(package_root))

    prefixes = tuple(
        f"{path.relative_to(repo_root).as_posix()}/" for path in package_roots
    )
    for relative in versioned_paths:
        if not _candidate_suffix(Path(relative)):
            continue
        if "/" not in relative or relative.startswith(prefixes):
            candidates.add(repo_root / relative)
    relatives = {path.relative_to(repo_root).as_posix() for path in candidates}
    ordered = _CONTROL_RELATIVES + tuple(
        sorted(relatives.difference(_CONTROL_RELATIVES))
    )
    casefolded: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for relative in ordered:
        previous = casefolded.setdefault(relative.casefold(), relative)
        if previous != relative:
            collisions.append((previous, relative))
    if collisions:
        _fail(
            "case_colliding_execution_surface",
            "case-colliding execution candidates are forbidden",
            collisions=tuple(collisions),
        )
    return (
        ordered,
        tuple(sorted(path.relative_to(repo_root).as_posix() for path in package_roots)),
        tuple(sorted(blocked)),
    )


def _git_blob_id(content: bytes, algorithm: str) -> str:
    if algorithm not in {"sha1", "sha256"}:
        _fail(
            "unsupported_git_object_format",
            "only SHA-1 and SHA-256 repositories are supported",
            object_format=algorithm,
        )
    digest = hashlib.new(algorithm)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def _worktree_identity(
    repo_root: Path,
    relative: str,
    algorithm: str,
) -> tuple[str, str, str, bool]:
    path = repo_root / relative
    try:
        content = path.read_bytes()
    except OSError as exc:
        _fail(
            "filesystem_inspection_failed",
            "an execution candidate could not be read",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )
    raw_sha = hashlib.sha256(content).hexdigest()
    crlf_equivalent = (
        relative in _CONTROL_RELATIVES
        or Path(relative).suffix.lower() in _CRLF_EQUIVALENT_SUFFIXES
    )
    compared = content.replace(b"\r\n", b"\n") if crlf_equivalent else content
    return (
        _git_blob_id(compared, algorithm),
        raw_sha,
        hashlib.sha256(compared).hexdigest(),
        compared != content,
    )


def _attest_candidates(
    executable: Path,
    repo_root: Path,
    candidates: tuple[str, ...],
) -> Mapping[str, Any]:
    algorithm = _git_text(
        executable, repo_root, ("rev-parse", "--show-object-format")
    )
    index = _parse_index(
        _run_git(executable, repo_root, ("ls-files", "--stage", "-z")).stdout
    )
    head = _parse_head(
        _run_git(executable, repo_root, ("ls-tree", "-r", "-z", "HEAD")).stdout
    )
    flags = _parse_flags(
        _run_git(executable, repo_root, ("ls-files", "-v", "-z")).stdout
    )
    violations: dict[str, list[Any]] = {
        "untracked_or_index_missing": [],
        "head_missing": [],
        "unmerged": [],
        "unsafe_index_flags": [],
        "head_index_divergence": [],
        "index_worktree_divergence": [],
        "worktree_missing": [],
        "non_blob_head_entries": [],
    }
    records: list[Mapping[str, Any]] = []
    normalized: list[str] = []
    for relative in candidates:
        index_entry = index.get(relative)
        head_entry = head.get(relative)
        if index_entry is None:
            violations["untracked_or_index_missing"].append(relative)
            continue
        if head_entry is None:
            violations["head_missing"].append(relative)
            continue
        index_mode, index_object, index_stage = index_entry
        head_mode, head_type, head_object = head_entry
        if index_stage != 0:
            violations["unmerged"].append((relative, index_stage))
        if flags.get(relative) != "H":
            violations["unsafe_index_flags"].append(
                (relative, flags.get(relative))
            )
        if head_type != "blob":
            violations["non_blob_head_entries"].append((relative, head_type))
        if (head_mode, head_object) != (index_mode, index_object):
            violations["head_index_divergence"].append(relative)
        path = repo_root / relative
        if not os.path.lexists(path):
            violations["worktree_missing"].append(relative)
            worktree_object = None
            raw_sha = None
            comparison_sha = None
            line_endings_normalized = False
        else:
            (
                worktree_object,
                raw_sha,
                comparison_sha,
                line_endings_normalized,
            ) = _worktree_identity(repo_root, relative, algorithm)
            if worktree_object != index_object:
                violations["index_worktree_divergence"].append(relative)
            if line_endings_normalized:
                normalized.append(relative)
        records.append(
            {
                "path": relative,
                "mode": index_mode,
                "blob": index_object,
                "worktree_blob": worktree_object,
                "worktree_raw_sha256": raw_sha,
                "worktree_comparison_sha256": comparison_sha,
                "line_endings_normalized": line_endings_normalized,
            }
        )
    populated = {key: values for key, values in violations.items() if values}
    if populated:
        _fail(
            "unsafe_execution_surface",
            "all execution candidates must match HEAD, index, and worktree",
            violations=populated,
        )
    canonical = json.dumps(
        records,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _freeze(
        {
            "object_format": algorithm,
            "candidate_count": len(candidates),
            "candidate_records_sha256": hashlib.sha256(canonical).hexdigest(),
            "untracked_candidates": 0,
            "head_index_divergences": 0,
            "index_worktree_divergences": 0,
            "line_ending_normalized_candidates": len(normalized),
            "line_ending_normalized_paths": tuple(normalized),
        }
    )


def _attest_preimport_environment(
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
    sys_path: Sequence[str],
    preloaded_agent_modules: Sequence[str],
    operation: str,
    stage: str,
) -> Mapping[str, Any]:
    root = _resolved_directory(repo_root, label="repository root")
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
    git = _find_git(root)
    top = Path(_git_text(git, root, ("rev-parse", "--show-toplevel"))).resolve()
    if top != root:
        _fail(
            "wrong_git_toplevel",
            "Git top level differs from the repository root",
            expected=str(root),
            actual=str(top),
        )
    head_commit = _git_text(git, root, ("rev-parse", "--verify", "HEAD"))
    head_tree = _git_text(git, root, ("rev-parse", "--verify", "HEAD^{tree}"))
    index = _parse_index(
        _run_git(git, root, ("ls-files", "--stage", "-z")).stdout
    )
    head = _parse_head(
        _run_git(git, root, ("ls-tree", "-r", "-z", "HEAD")).stdout
    )
    versioned_paths = tuple(sorted(set(index) | set(head)))
    candidates, package_roots, blocked = _execution_surface(
        root, versioned_paths
    )
    if _BOOTSTRAP_RELATIVE_PATH.as_posix() not in candidates:
        _fail(
            "bootstrap_outside_execution_surface",
            "bootstrap is absent from the attested execution surface",
        )
    candidate_evidence = _attest_candidates(git, root, candidates)
    body: dict[str, Any] = {
        "attestation_schema_version": ATTESTATION_SCHEMA_VERSION,
        "bootstrap_id": BOOTSTRAP_ID,
        "invocation": {"operation": operation, "stage": stage},
        "repository": {
            "root": str(root),
            "git_executable": str(git),
            "head_commit": head_commit,
            "head_tree": head_tree,
        },
        "runtime": runtime,
        "execution_surface": {
            "candidate_suffixes": IMPORT_CANDIDATE_SUFFIXES,
            "candidate_paths": candidates,
            "package_roots": package_roots,
            "blocked_root_imports": blocked,
            **_thaw(candidate_evidence),
        },
        "checks": {
            "fixed_argv": True,
            "runtime_isolated_before_import": True,
            "cwd_is_repo_root": True,
            "all_import_candidates_head_index_worktree_equal": True,
            "no_import_candidate_untracked": True,
            "no_execution_surface_symlink_or_reparse": True,
            "dispatch_environment_sanitized": True,
            "dispatch_sys_path_sanitized": True,
            "unattested_root_imports_blocked": True,
        },
    }
    canonical = json.dumps(
        _thaw(body),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    body["attestation_sha256"] = hashlib.sha256(canonical).hexdigest()
    return _freeze(body)


def require_active_attestation(
    *,
    operation: str,
    stage: str,
    repo_root: Path,
) -> Mapping[str, Any]:
    """Require the process-local proof installed only during guarded dispatch."""

    value = getattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE, None)
    if not isinstance(value, Mapping):
        _fail(
            "missing_active_attestation",
            "audit runner and verifier require the isolated audit bootstrap",
        )
    thawed = _thaw(value)
    supplied_hash = thawed.pop("attestation_sha256", None)
    canonical = json.dumps(
        thawed,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    expected_root = str(_resolved_directory(repo_root, label="repository root"))
    if (
        operation not in ALLOWED_OPERATIONS
        or stage != "audit"
        or supplied_hash != hashlib.sha256(canonical).hexdigest()
        or thawed.get("attestation_schema_version")
        != ATTESTATION_SCHEMA_VERSION
        or thawed.get("bootstrap_id") != BOOTSTRAP_ID
        or thawed.get("invocation")
        != {"operation": operation, "stage": stage}
        or not isinstance(thawed.get("repository"), dict)
        or thawed["repository"].get("root") != expected_root
        or sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or not sys.dont_write_bytecode
    ):
        _fail(
            "invalid_active_attestation",
            "active bootstrap evidence does not match this audit invocation",
            operation=operation,
            stage=stage,
            repo_root=expected_root,
        )
    return value


def _import_dispatch(operation: str, stage: str) -> tuple[Mapping[str, Any], int]:
    module_name, callable_name = _DISPATCH_TARGETS[operation]
    module = importlib.import_module(module_name)
    target = getattr(module, callable_name, None)
    if not callable(target):
        _fail(
            "dispatch_target_missing",
            "the fixed audit dispatch target is unavailable",
            module=module_name,
            callable=callable_name,
        )
    result = target(stage)
    if not isinstance(result, Mapping):
        _fail(
            "invalid_dispatch_result",
            "the fixed audit dispatch target did not return a mapping",
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
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            continue
        if resolved != repo_root and not _path_is_within(resolved, repo_root):
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
) -> tuple[Mapping[str, Any], int]:
    operation, stage = _parse_fixed_argv(argv)
    attestation = _attest_preimport_environment(
        repo_root=repo_root,
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
    )
    if hasattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE):
        _fail(
            "nested_active_attestation",
            "an audit bootstrap attestation is already active",
        )
    setattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE, attestation)
    original_environment = dict(os.environ)
    original_sys_path = list(sys.path)
    original_meta_path = list(sys.meta_path)
    sanitized_environment = _sanitized_environment()
    blocker = _BlockedRootImportFinder(
        attestation["execution_surface"]["blocked_root_imports"]
    )
    os.environ.clear()
    os.environ.update(sanitized_environment)
    sys.path[:] = _safe_dispatch_sys_path(repo_root)
    sys.meta_path.insert(0, blocker)
    try:
        result, exit_code = dispatch(operation, stage)
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
                "result": result,
            }
        ),
        exit_code,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Attest the execution surface, then invoke one fixed audit callable."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    payload, exit_code = _guarded_main(
        arguments,
        repo_root=_repo_root_from_file(),
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
    "BOOTSTRAP_ID",
    "BootstrapSecurityError",
    "DispatchCallable",
    "IMPORT_CANDIDATE_SUFFIXES",
    "main",
    "require_active_attestation",
]
