"""Isolated pre-import bootstrap for contextual aggregation stages.

The only supported command shape is::

    python -I -B agent_benchmark/contextual_expert_aggregation_bootstrap.py \
        {stage|verify} {development|confirmation}

This file intentionally imports only the Python standard library at module
load time.  The repository root is not placed on ``sys.path`` and no
``agent_benchmark`` package is imported until the complete execution surface
has been proved to match both ``HEAD`` and the index.  Python and frozen text
control files permit only an internally implemented CRLF-to-LF equivalence;
native and binary candidates require literal byte identity.
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
BOOTSTRAP_ID = "contextual-expert-aggregation-isolated-bootstrap-v1"
ALLOWED_OPERATIONS = ("stage", "verify")
ALLOWED_STAGES = ("development", "confirmation")
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
    "agent_benchmark/contextual_expert_aggregation_bootstrap.py"
)
_APPROVED_INFO_EXCLUDE_LINES = (".codex/",)
_ACTIVE_ATTESTATION_ATTRIBUTE = (
    "_contextual_expert_aggregation_bootstrap_attestation_v1"
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
_CONTROL_CANDIDATE_PATHS = (
    Path(".gitattributes"),
    Path(".gitignore"),
    Path("requirements.txt"),
)
_CONTROL_CANDIDATE_RELATIVES = tuple(
    path.as_posix() for path in _CONTROL_CANDIDATE_PATHS
)
_CRLF_EQUIVALENT_SUFFIXES = frozenset({".py", ".pyw"})
_DANGEROUS_GIT_ATTRIBUTES = frozenset(
    {"filter", "ident", "working-tree-encoding"}
)
_DISPATCH_TARGETS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "stage": (
            "agent_benchmark.contextual_expert_aggregation_stage",
            "run_stage",
        ),
        "verify": (
            "agent_benchmark.contextual_expert_aggregation_verifier",
            "verify_stage",
        ),
    }
)

DispatchCallable = Callable[[str, str], tuple[Mapping[str, Any], int]]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
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
    """Fail-closed bootstrap rejection with machine-readable evidence."""

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


def _fail(
    code: str,
    message: str,
    **details: Any,
) -> None:
    raise BootstrapSecurityError(code, message, details=details)


def _parse_fixed_argv(argv: Sequence[str]) -> tuple[str, str]:
    values = tuple(argv)
    if len(values) != 2:
        _fail(
            "unsafe_argv",
            "exactly two positional arguments are required",
            expected=("{stage|verify}", "{development|confirmation}"),
            received=values,
        )
    operation, stage = values
    if operation not in ALLOWED_OPERATIONS or stage not in ALLOWED_STAGES:
        _fail(
            "unsafe_argv",
            "operation or stage is outside the fixed command surface",
            allowed_operations=ALLOWED_OPERATIONS,
            allowed_stages=ALLOWED_STAGES,
            received=values,
        )
    return operation, stage


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        _fail(
            "filesystem_inspection_failed",
            "could not inspect an execution-surface path",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )


def _is_link_or_reparse(path: Path) -> bool:
    info = _lstat(path)
    return stat.S_ISLNK(info.st_mode) or _is_reparse(info)


def _resolved_directory(path: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        _fail(
            "invalid_directory",
            f"{label} cannot be resolved",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )
    if not resolved.is_dir():
        _fail("invalid_directory", f"{label} is not a directory", path=str(resolved))
    return resolved


def _repo_root_from_file() -> Path:
    source = Path(os.path.abspath(__file__))
    if _is_link_or_reparse(source):
        _fail(
            "foreign_bootstrap_location",
            "bootstrap source is a symlink or reparse point",
            source=str(source),
        )
    resolved_source = source.resolve(strict=True)
    if source != resolved_source:
        _fail(
            "foreign_bootstrap_location",
            "bootstrap source path is not canonical",
            source=str(source),
            resolved_source=str(resolved_source),
        )
    root = resolved_source.parent.parent
    expected = root / _BOOTSTRAP_RELATIVE_PATH
    if resolved_source != expected.resolve(strict=True):
        _fail(
            "foreign_bootstrap_location",
            "bootstrap source is not at its fixed repository-relative path",
            resolved_source=str(resolved_source),
            expected_source=str(expected),
        )
    return _resolved_directory(root, label="repository root derived from __file__")


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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
            isolated=int(isolated),
        )
    if int(dont_write_bytecode_flag) != 1 or not bool(dont_write_bytecode_value):
        _fail(
            "bytecode_writes_disabled_required",
            "Python bytecode writes must be disabled; invoke with -B",
            dont_write_bytecode_flag=int(dont_write_bytecode_flag),
            dont_write_bytecode_value=bool(dont_write_bytecode_value),
        )
    expected_tail = (
        "-I",
        "-B",
        _BOOTSTRAP_RELATIVE_PATH.as_posix(),
        operation,
        stage,
    )
    observed_argv = tuple(orig_argv)
    if (
        int(optimize) != 0
        or len(observed_argv) < 2
        or observed_argv[1:] != expected_tail
        or tuple(warn_options)
        or dict(xoptions)
    ):
        _fail(
            "noncanonical_interpreter_flags",
            "interpreter flags must exactly match the frozen -I -B command",
            optimize=int(optimize),
            expected_tail=expected_tail,
            observed_argv=observed_argv,
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

    unsafe_entries: list[str] = []
    relative_entries: list[str] = []
    for raw_entry in sys_path:
        if not raw_entry:
            relative_entries.append(raw_entry)
            continue
        entry = Path(raw_entry)
        if not entry.is_absolute():
            relative_entries.append(raw_entry)
            continue
        try:
            resolved_entry = entry.resolve(strict=False)
        except OSError:
            resolved_entry = entry.absolute()
        if resolved_entry == repo_root or _path_is_within(resolved_entry, repo_root):
            unsafe_entries.append(raw_entry)
    if relative_entries or unsafe_entries:
        _fail(
            "unsafe_preimport_sys_path",
            "repository or relative paths are present before attestation",
            relative_entries=tuple(relative_entries),
            repository_entries=tuple(unsafe_entries),
        )

    preloaded = tuple(sorted(preloaded_agent_modules))
    if preloaded:
        _fail(
            "package_imported_before_attestation",
            "agent_benchmark was imported before the execution surface was attested",
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
            "incoming_git_environment_keys": tuple(
                sorted(
                    key for key in os.environ if key.upper().startswith("GIT_")
                )
            ),
            "dispatch_git_environment_sanitized": True,
            "cwd_equals_repository_root": True,
            "preimport_sys_path_clean": True,
            "agent_package_preloaded": False,
        }
    )


def _find_git_executable(repo_root: Path) -> Path:
    found = shutil.which("git")
    if not found:
        _fail("git_unavailable", "Git executable was not found on PATH")
    executable = Path(found).resolve(strict=True)
    if executable == repo_root or _path_is_within(executable, repo_root):
        _fail(
            "foreign_git_executable",
            "Git executable resolves inside the repository",
            executable=str(executable),
        )
    return executable


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_ATTR_SOURCE": "HEAD",
            "GIT_CONFIG_COUNT": "4",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_KEY_0": "core.attributesFile",
            "GIT_CONFIG_KEY_1": "core.excludesFile",
            "GIT_CONFIG_KEY_2": "core.autocrlf",
            "GIT_CONFIG_KEY_3": "core.fsmonitor",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_VALUE_0": os.devnull,
            "GIT_CONFIG_VALUE_1": os.devnull,
            "GIT_CONFIG_VALUE_2": "true",
            "GIT_CONFIG_VALUE_3": "false",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _run_git(
    executable: Path,
    repo_root: Path,
    args: Sequence[str],
    *,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            [str(executable), "-C", str(repo_root), *args],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_git_environment(),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(
            "git_inspection_failed",
            "Git inspection command could not run",
            args=tuple(args),
            error=f"{type(exc).__name__}: {exc}",
        )
    if check and completed.returncode != 0:
        _fail(
            "git_inspection_failed",
            "Git inspection command failed",
            args=tuple(args),
            returncode=completed.returncode,
            stderr=completed.stderr.decode("utf-8", errors="replace").strip(),
        )
    return completed


def _git_text(
    executable: Path,
    repo_root: Path,
    args: Sequence[str],
) -> str:
    return _run_git(executable, repo_root, args).stdout.decode(
        "utf-8", errors="strict"
    ).strip()


def _effective_lines(path: Path) -> tuple[str, ...]:
    if not os.path.lexists(path):
        return ()
    if _is_link_or_reparse(path):
        _fail(
            "unsafe_git_metadata",
            "local Git control file is a symlink or reparse point",
            path=str(path),
        )
    try:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
    except (OSError, UnicodeError) as exc:
        _fail(
            "unsafe_git_metadata",
            "local Git control file cannot be read as UTF-8",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )
    return tuple(
        line
        for line in text.splitlines()
        if line != "" and not line.startswith("#")
    )


def _validate_local_git_controls(
    executable: Path,
    repo_root: Path,
) -> Mapping[str, Any]:
    git_dir_text = _git_text(executable, repo_root, ("rev-parse", "--absolute-git-dir"))
    git_dir = Path(git_dir_text).resolve(strict=True)

    control_paths = {
        "info_exclude": Path(
            _git_text(executable, repo_root, ("rev-parse", "--git-path", "info/exclude"))
        ),
        "info_attributes": Path(
            _git_text(
                executable,
                repo_root,
                ("rev-parse", "--git-path", "info/attributes"),
            )
        ),
        "info_grafts": Path(
            _git_text(executable, repo_root, ("rev-parse", "--git-path", "info/grafts"))
        ),
        "object_alternates": Path(
            _git_text(
                executable,
                repo_root,
                ("rev-parse", "--git-path", "objects/info/alternates"),
            )
        ),
    }
    for name, path in tuple(control_paths.items()):
        if not path.is_absolute():
            control_paths[name] = repo_root / path

    info_exclude_lines = _effective_lines(control_paths["info_exclude"])
    effective_controls: dict[str, tuple[str, ...]] = {}
    if info_exclude_lines not in ((), _APPROVED_INFO_EXCLUDE_LINES):
        effective_controls["info_exclude"] = info_exclude_lines
    for name, path in control_paths.items():
        if name == "info_exclude":
            continue
        lines = _effective_lines(path)
        if lines:
            effective_controls[name] = lines
    if effective_controls:
        _fail(
            "unsafe_local_git_controls",
            "local Git exclude, attribute, graft, or alternate-object rules are forbidden",
            controls=effective_controls,
        )

    config = _run_git(
        executable,
        repo_root,
        (
            "config",
            "--local",
            "--null",
            "--get-regexp",
            r"^(filter\..*\.(clean|process|required)|core\.(autocrlf|eol|safecrlf|excludesfile|attributesfile|sparsecheckout|sparsecheckoutcone|fsmonitor)|include(if\..*)?\.path|extensions\.worktreeconfig)$",
        ),
        check=False,
    )
    if config.returncode not in (0, 1):
        _fail(
            "git_inspection_failed",
            "local Git configuration could not be inspected",
            returncode=config.returncode,
            stderr=config.stderr.decode("utf-8", errors="replace").strip(),
        )
    config_entries = tuple(
        item.decode("utf-8", errors="strict")
        for item in config.stdout.split(b"\0")
        if item
    )
    if config_entries:
        _fail(
            "unsafe_local_git_controls",
            "local Git configuration changes filter, include, exclude, sparse, or executable monitor semantics",
            config_entries=config_entries,
        )

    replacements = _git_text(executable, repo_root, ("replace", "--list"))
    if replacements:
        _fail(
            "unsafe_local_git_controls",
            "Git replacement objects are forbidden",
            replacement_refs=tuple(replacements.splitlines()),
        )

    return _freeze(
        {
            "git_dir": str(git_dir),
            "approved_info_exclude_lines": info_exclude_lines,
            "info_attributes_effective_lines": 0,
            "info_grafts_effective_lines": 0,
            "object_alternates_effective_lines": 0,
            "dangerous_local_config_entries": 0,
            "replacement_refs": 0,
            "external_config_files_disabled": True,
            "external_attribute_files_disabled": True,
            "worktree_attributes_replaced_by_head": True,
            "command_scope_core_autocrlf": True,
            "command_scope_core_fsmonitor": False,
        }
    )


def _validate_head_git_attributes(
    executable: Path,
    repo_root: Path,
) -> Mapping[str, Any]:
    head = _parse_head_entries(
        _run_git(executable, repo_root, ("ls-tree", "-r", "-z", "HEAD")).stdout
    )
    attribute_paths = tuple(
        sorted(
            relative
            for relative in head
            if relative == ".gitattributes" or relative.endswith("/.gitattributes")
        )
    )
    records: list[Mapping[str, Any]] = []
    dangerous: list[Mapping[str, Any]] = []
    for relative in attribute_paths:
        _mode, object_type, object_id = head[relative]
        if object_type != "blob":
            _fail(
                "unsafe_git_attributes",
                "a committed .gitattributes entry is not a blob",
                path=relative,
                object_type=object_type,
            )
        content = _run_git(
            executable,
            repo_root,
            ("cat-file", "blob", object_id),
        ).stdout
        try:
            text = content.decode("utf-8-sig", errors="strict")
        except UnicodeError as exc:
            _fail(
                "unsafe_git_attributes",
                "a committed .gitattributes file is not UTF-8 text",
                path=relative,
                error=f"{type(exc).__name__}: {exc}",
            )
        effective_lines = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            effective_lines += 1
            fields = stripped.split()
            for field in fields[1:]:
                attribute = field
                if attribute.startswith(("-", "!")):
                    attribute = attribute[1:]
                attribute = attribute.split("=", 1)[0]
                if attribute in _DANGEROUS_GIT_ATTRIBUTES:
                    dangerous.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "attribute": attribute,
                        }
                    )
        records.append(
            {
                "path": relative,
                "blob": object_id,
                "sha256": hashlib.sha256(content).hexdigest(),
                "effective_lines": effective_lines,
            }
        )
    if dangerous:
        _fail(
            "unsafe_git_attributes",
            "committed attributes may not select filters, ident expansion, or working-tree transcoding",
            occurrences=dangerous,
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
            "source": "HEAD",
            "attribute_file_count": len(attribute_paths),
            "attribute_records_sha256": hashlib.sha256(canonical).hexdigest(),
            "dangerous_attribute_occurrences": 0,
        }
    )


def _candidate_suffix(path: Path) -> bool:
    return path.suffix.lower() in IMPORT_CANDIDATE_SUFFIXES


def _scan_tree(directory: Path, repo_root: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    stack = [directory]
    while stack:
        current = stack.pop()
        if _is_link_or_reparse(current):
            _fail(
                "unsafe_execution_surface_link",
                "an import-surface directory is a symlink or reparse point",
                path=str(current),
            )
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name.casefold())
        except OSError as exc:
            _fail(
                "filesystem_inspection_failed",
                "could not enumerate an import-surface directory",
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
                    "could not inspect an import-surface entry",
                    path=str(path),
                    error=f"{type(exc).__name__}: {exc}",
                )
            linked = entry.is_symlink() or _is_reparse(info)
            if linked:
                _fail(
                    "unsafe_execution_surface_link",
                    "an import-surface entry is a symlink or reparse point",
                    path=str(path),
                )
            if stat.S_ISDIR(info.st_mode):
                stack.append(path)
            elif _candidate_suffix(path):
                if not stat.S_ISREG(info.st_mode):
                    _fail(
                        "unsafe_execution_surface_type",
                        "an import candidate is not a regular file",
                        path=str(path),
                    )
                candidates.append(path)
    return tuple(candidates)


def _execution_surface(
    repo_root: Path,
    versioned_paths: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    candidates: set[Path] = set()
    package_roots: set[Path] = set()
    blocked_root_imports: set[str] = set()
    agent_root = repo_root / _AGENT_PACKAGE
    if not agent_root.exists() or not agent_root.is_dir():
        _fail(
            "missing_agent_package",
            "agent_benchmark package directory is missing",
            path=str(agent_root),
        )

    try:
        root_entries = sorted(os.scandir(repo_root), key=lambda item: item.name.casefold())
    except OSError as exc:
        _fail(
            "filesystem_inspection_failed",
            "could not enumerate repository root",
            error=f"{type(exc).__name__}: {exc}",
        )
    for entry in root_entries:
        path = Path(entry.path)
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            _fail(
                "filesystem_inspection_failed",
                "could not inspect a repository-root entry",
                path=str(path),
                error=f"{type(exc).__name__}: {exc}",
            )
        if entry.is_symlink() or _is_reparse(info):
            _fail(
                "unsafe_execution_surface_link",
                "a repository-root entry is a symlink or reparse point",
                path=str(path),
            )
        if stat.S_ISREG(info.st_mode) and _candidate_suffix(path):
            candidates.add(path)
        elif stat.S_ISDIR(info.st_mode):
            if path == agent_root:
                package_roots.add(path)
            elif entry.name in _BLOCKED_ROOT_IMPORTS:
                blocked_root_imports.add(entry.name)
            else:
                package_roots.add(path)

    candidates.update(repo_root / path for path in _CONTROL_CANDIDATE_PATHS)

    package_roots.add(agent_root)
    versioned_package_names = {
        relative.split("/", 1)[0]
        for relative in versioned_paths
        if relative.count("/") == 1
        and relative.split("/", 1)[1] == "__init__.py"
    }
    for package_name in versioned_package_names:
        if package_name not in blocked_root_imports:
            package_roots.add(repo_root / package_name)

    for package_root in sorted(package_roots, key=lambda path: str(path).casefold()):
        if package_root.exists():
            if not package_root.is_dir():
                _fail(
                    "unsafe_execution_surface_type",
                    "a versioned root package is not a directory in the worktree",
                    path=str(package_root),
                )
            candidates.update(_scan_tree(package_root, repo_root))

    package_prefixes = tuple(
        f"{path.relative_to(repo_root).as_posix()}/" for path in package_roots
    )
    for relative in versioned_paths:
        relative_path = Path(relative)
        if not _candidate_suffix(relative_path):
            continue
        if "/" not in relative or relative.startswith(package_prefixes):
            candidates.add(repo_root / Path(relative))

    discovered_relatives = {
        path.relative_to(repo_root).as_posix() for path in candidates
    }
    relative_candidates = _CONTROL_CANDIDATE_RELATIVES + tuple(
        sorted(discovered_relatives.difference(_CONTROL_CANDIDATE_RELATIVES))
    )
    relative_packages = tuple(
        sorted(path.relative_to(repo_root).as_posix() for path in package_roots)
    )
    casefolded: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []
    for relative in relative_candidates:
        previous = casefolded.setdefault(relative.casefold(), relative)
        if previous != relative:
            collisions.append((previous, relative))
    if collisions:
        _fail(
            "case_colliding_execution_surface",
            "case-colliding import candidates are forbidden",
            collisions=tuple(collisions),
        )
    return (
        relative_candidates,
        relative_packages,
        tuple(sorted(blocked_root_imports)),
    )


def _versioned_paths(
    executable: Path,
    repo_root: Path,
) -> tuple[str, ...]:
    index = _parse_index_entries(
        _run_git(executable, repo_root, ("ls-files", "--stage", "-z")).stdout
    )
    head = _parse_head_entries(
        _run_git(executable, repo_root, ("ls-tree", "-r", "-z", "HEAD")).stdout
    )
    return tuple(sorted(set(index) | set(head)))


def _parse_index_entries(raw: bytes) -> dict[str, tuple[str, str, int]]:
    entries: dict[str, tuple[str, str, int]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, path_bytes = record.partition(b"\t")
        if not separator:
            _fail("git_parse_failed", "malformed index entry")
        parts = metadata.decode("ascii", errors="strict").split()
        if len(parts) != 3:
            _fail("git_parse_failed", "malformed index metadata")
        mode, object_id, stage_text = parts
        path = path_bytes.decode("utf-8", errors="surrogateescape")
        stage_number = int(stage_text)
        if path in entries:
            _fail(
                "unmerged_execution_surface",
                "multiple index stages exist for an import candidate",
                path=path,
            )
        entries[path] = (mode, object_id, stage_number)
    return entries


def _parse_head_entries(raw: bytes) -> dict[str, tuple[str, str, str]]:
    entries: dict[str, tuple[str, str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, path_bytes = record.partition(b"\t")
        if not separator:
            _fail("git_parse_failed", "malformed HEAD tree entry")
        parts = metadata.decode("ascii", errors="strict").split()
        if len(parts) != 3:
            _fail("git_parse_failed", "malformed HEAD tree metadata")
        mode, object_type, object_id = parts
        path = path_bytes.decode("utf-8", errors="surrogateescape")
        entries[path] = (mode, object_type, object_id)
    return entries


def _parse_index_flags(raw: bytes) -> dict[str, str]:
    flags: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        text = record.decode("utf-8", errors="surrogateescape")
        marker, separator, path = text.partition(" ")
        if not separator or len(marker) != 1:
            _fail("git_parse_failed", "malformed index flag entry")
        flags[path] = marker
    return flags


def _git_blob_id(content: bytes, algorithm: str) -> str:
    if algorithm not in {"sha1", "sha256"}:
        _fail(
            "unsupported_git_object_format",
            "only SHA-1 and SHA-256 Git repositories are supported",
            object_format=algorithm,
        )
    digest = hashlib.new(algorithm)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def _worktree_blob_identity(
    repo_root: Path, relative: str, algorithm: str
) -> tuple[str, str, str, bool]:
    path = repo_root / relative
    try:
        content = path.read_bytes()
    except OSError as exc:
        _fail(
            "filesystem_inspection_failed",
            "could not read an import candidate",
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )
    raw_sha256 = hashlib.sha256(content).hexdigest()
    crlf_equivalent = (
        relative in _CONTROL_CANDIDATE_RELATIVES
        or Path(relative).suffix.lower() in _CRLF_EQUIVALENT_SUFFIXES
    )
    comparison_content = content.replace(b"\r\n", b"\n") if crlf_equivalent else content
    comparison_sha256 = hashlib.sha256(comparison_content).hexdigest()
    return (
        _git_blob_id(comparison_content, algorithm),
        raw_sha256,
        comparison_sha256,
        comparison_content != content,
    )


def _raise_candidate_violations(violations: Mapping[str, Sequence[Any]]) -> None:
    populated = {key: list(value) for key, value in violations.items() if value}
    if populated:
        _fail(
            "unsafe_execution_surface",
            "execution candidates must be tracked, unignored, and identical in HEAD, index, and worktree",
            violations=populated,
        )


def _attest_candidates(
    executable: Path,
    repo_root: Path,
    candidates: tuple[str, ...],
) -> Mapping[str, Any]:
    object_format = _git_text(
        executable, repo_root, ("rev-parse", "--show-object-format")
    )
    if object_format not in {"sha1", "sha256"}:
        _fail(
            "unsupported_git_object_format",
            "only SHA-1 and SHA-256 Git repositories are supported",
            object_format=object_format,
        )
    index = _parse_index_entries(
        _run_git(executable, repo_root, ("ls-files", "--stage", "-z")).stdout
    )
    head = _parse_head_entries(
        _run_git(executable, repo_root, ("ls-tree", "-r", "-z", "HEAD")).stdout
    )
    flags = _parse_index_flags(
        _run_git(executable, repo_root, ("ls-files", "-v", "-z")).stdout
    )

    encoded_paths = b"".join(
        relative.encode("utf-8", errors="surrogateescape") + b"\0"
        for relative in candidates
    )
    ignored_process = _run_git(
        executable,
        repo_root,
        ("check-ignore", "--no-index", "-z", "--stdin"),
        input_bytes=encoded_paths,
        check=False,
    )
    if ignored_process.returncode not in (0, 1):
        _fail(
            "git_inspection_failed",
            "Git ignore rules could not be inspected",
            returncode=ignored_process.returncode,
            stderr=ignored_process.stderr.decode("utf-8", errors="replace").strip(),
        )
    ignored = tuple(
        item.decode("utf-8", errors="surrogateescape")
        for item in ignored_process.stdout.split(b"\0")
        if item
    )
    ignored_set = frozenset(ignored)

    violations: dict[str, list[Any]] = {
        "ignored": [],
        "untracked_or_index_missing": [],
        "head_missing": [],
        "unmerged": [],
        "unsafe_index_flags": [],
        "head_index_divergence": [],
        "index_worktree_divergence": [],
        "non_blob_head_entries": [],
    }
    records: list[Mapping[str, Any]] = []
    normalized_paths: list[str] = []

    def attest_candidate(relative: str) -> None:
        if relative in ignored_set:
            violations["ignored"].append(relative)
        index_entry = index.get(relative)
        head_entry = head.get(relative)
        if index_entry is None:
            violations["untracked_or_index_missing"].append(relative)
            return
        if head_entry is None:
            violations["head_missing"].append(relative)
            return
        index_mode, index_object, index_stage = index_entry
        head_mode, head_type, head_object = head_entry
        if index_stage != 0:
            violations["unmerged"].append((relative, index_stage))
        marker = flags.get(relative)
        if marker != "H":
            violations["unsafe_index_flags"].append((relative, marker))
        if head_type != "blob":
            violations["non_blob_head_entries"].append((relative, head_type))
        if (head_mode, head_object) != (index_mode, index_object):
            violations["head_index_divergence"].append(relative)
        worktree_path = repo_root / relative
        if not os.path.lexists(worktree_path):
            violations.setdefault("worktree_missing", []).append(relative)
            worktree_object = None
            worktree_raw_sha256 = None
            worktree_comparison_sha256 = None
            line_endings_normalized = False
        else:
            (
                worktree_object,
                worktree_raw_sha256,
                worktree_comparison_sha256,
                line_endings_normalized,
            ) = _worktree_blob_identity(
                repo_root, relative, object_format
            )
            if line_endings_normalized:
                normalized_paths.append(relative)
            if worktree_object != index_object:
                violations["index_worktree_divergence"].append(relative)
        records.append(
            {
                "path": relative,
                "mode": index_mode,
                "blob": index_object,
                "worktree_blob": worktree_object,
                "worktree_raw_sha256": worktree_raw_sha256,
                "worktree_comparison_sha256": worktree_comparison_sha256,
                "line_endings_normalized": line_endings_normalized,
            }
        )

    control_candidates = tuple(
        relative for relative in candidates if relative in _CONTROL_CANDIDATE_RELATIVES
    )
    runtime_candidates = tuple(
        relative for relative in candidates if relative not in _CONTROL_CANDIDATE_RELATIVES
    )
    for relative in control_candidates:
        attest_candidate(relative)
    _raise_candidate_violations(violations)
    for relative in runtime_candidates:
        attest_candidate(relative)
    _raise_candidate_violations(violations)

    canonical_records = json.dumps(
        records,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _freeze(
        {
            "object_format": object_format,
            "candidate_count": len(candidates),
            "candidate_records_sha256": hashlib.sha256(canonical_records).hexdigest(),
            "ignored_candidates": 0,
            "untracked_candidates": 0,
            "head_index_divergences": 0,
            "index_worktree_divergences": 0,
            "line_ending_normalized_candidates": len(normalized_paths),
            "line_ending_normalized_paths": tuple(normalized_paths),
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
    git_executable = _find_git_executable(root)
    top_level = Path(
        _git_text(git_executable, root, ("rev-parse", "--show-toplevel"))
    ).resolve(strict=True)
    if top_level != root:
        _fail(
            "wrong_git_toplevel",
            "Git top-level directory does not equal the repository root",
            expected=str(root),
            actual=str(top_level),
        )
    head_commit = _git_text(git_executable, root, ("rev-parse", "--verify", "HEAD"))
    head_tree = _git_text(
        git_executable, root, ("rev-parse", "--verify", "HEAD^{tree}")
    )
    local_controls = _validate_local_git_controls(git_executable, root)
    versioned_paths = _versioned_paths(git_executable, root)
    candidates, package_roots, blocked_root_imports = _execution_surface(
        root,
        versioned_paths,
    )
    if _BOOTSTRAP_RELATIVE_PATH.as_posix() not in candidates:
        _fail(
            "bootstrap_outside_execution_surface",
            "bootstrap source was not found in the attested candidate set",
            expected=_BOOTSTRAP_RELATIVE_PATH.as_posix(),
        )
    candidate_evidence = _attest_candidates(
        git_executable, root, candidates
    )
    head_git_attributes = _validate_head_git_attributes(git_executable, root)

    body = {
        "attestation_schema_version": ATTESTATION_SCHEMA_VERSION,
        "bootstrap_id": BOOTSTRAP_ID,
        "invocation": {"operation": operation, "stage": stage},
        "repository": {
            "root": str(root),
            "git_executable": str(git_executable),
            "head_commit": head_commit,
            "head_tree": head_tree,
        },
        "runtime": runtime,
        "local_git_controls": local_controls,
        "head_git_attributes": head_git_attributes,
        "execution_surface": {
            "candidate_suffixes": IMPORT_CANDIDATE_SUFFIXES,
            "package_roots": package_roots,
            "blocked_root_imports": blocked_root_imports,
            "candidate_paths": candidates,
            **_thaw(candidate_evidence),
        },
        "checks": {
            "fixed_argv": True,
            "runtime_isolated_before_import": True,
            "cwd_is_repo_root": True,
            "local_git_controls_safe": True,
            "head_git_attributes_safe": True,
            "all_import_candidates_head_index_worktree_equal": True,
            "no_import_candidate_ignored_or_untracked": True,
            "no_execution_surface_symlink_or_reparse": True,
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
    *, operation: str, stage: str, repo_root: Path
) -> Mapping[str, Any]:
    """Require the process-only proof installed around guarded dispatch."""

    value = getattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE, None)
    if not isinstance(value, Mapping):
        _fail(
            "missing_active_attestation",
            "stage and verifier entry require the isolated bootstrap",
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
        supplied_hash != hashlib.sha256(canonical).hexdigest()
        or thawed.get("attestation_schema_version") != ATTESTATION_SCHEMA_VERSION
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
            "active bootstrap evidence does not match this invocation",
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
            "fixed dispatch callable is unavailable",
            module=module_name,
            callable=callable_name,
        )
    result = target(stage)
    if not isinstance(result, Mapping):
        _fail(
            "invalid_dispatch_result",
            "fixed dispatch callable did not return a mapping",
            module=module_name,
            callable=callable_name,
            result_type=type(result).__name__,
        )
    exit_code = 0
    if operation == "stage" and not bool(result.get("stage_pass")):
        exit_code = 2
    return result, exit_code


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

    root_text = str(attestation["repository"]["root"])
    if hasattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE):
        _fail(
            "nested_active_attestation",
            "an active bootstrap attestation already exists",
        )
    setattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE, attestation)
    original_environment = dict(os.environ)
    sanitized_environment = _git_environment()
    os.environ.clear()
    os.environ.update(sanitized_environment)
    import_blocker = _BlockedRootImportFinder(
        attestation["execution_surface"]["blocked_root_imports"]
    )
    sys.meta_path.insert(0, import_blocker)
    sys.path.insert(0, root_text)
    try:
        result, exit_code = dispatch(operation, stage)
    finally:
        if sys.path and sys.path[0] == root_text:
            del sys.path[0]
        else:  # pragma: no cover - defensive against a dispatch mutating sys.path
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass
        try:
            delattr(sys, _ACTIVE_ATTESTATION_ATTRIBUTE)
        except AttributeError:  # pragma: no cover - defensive against dispatch
            pass
        os.environ.clear()
        os.environ.update(original_environment)
        try:
            sys.meta_path.remove(import_blocker)
        except ValueError:  # pragma: no cover - defensive against dispatch
            pass
    if not isinstance(result, Mapping):
        _fail(
            "invalid_dispatch_result",
            "guarded dispatch did not return a mapping",
            result_type=type(result).__name__,
        )
    if (
        not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or not 0 <= exit_code <= 255
    ):
        _fail(
            "invalid_dispatch_exit_code",
            "guarded dispatch did not return an integer exit code",
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
    """Attest the repository, then invoke exactly one fixed callable."""

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
    "IMPORT_CANDIDATE_SUFFIXES",
    "main",
    "require_active_attestation",
]
