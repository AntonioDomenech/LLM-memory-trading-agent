"""Crash-recoverable v2.2 publication through one stable annotated Git tag.

This module deliberately separates deterministic publication preparation from
remote transport.  The caller must durably commit the publication intent,
transport manifest, worker ownership, remote observations, and one-use push
authorization through the append-only store before advancing between phases.

The publisher never reads Git configuration from the implementation
repository.  It uses a fresh bare Git directory with one exact object
alternate, an empty hooks directory, a literal HTTPS endpoint, absolute pinned
executables, and a newly constructed child environment.  Remote command
execution is injected because the runner owns the Windows Job Object that
contains and kills the complete Git/helper process tree.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Callable, Final, Mapping
import zlib

from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    validate_implementation_manifest,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    DEVELOPMENT_ATTEMPT_ID,
    EMPTY_BYTES_SHA256,
    EXTERNAL_PUBLICATION_FIELDS,
    EXTERNAL_TAG_MESSAGE_FIELDS,
    EXTERNAL_TAG_REF_TEMPLATE,
    FINAL_ATTEMPT_ID,
    FINAL_REGISTRY_TAG_REF_TEMPLATE,
    PUBLICATION_INTENT_FIELDS,
    PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256,
    PUBLICATION_NORMAL_OPERATION_SHA256,
    PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS,
    PUBLICATION_PUSH_COMMAND_PROFILE_SHA256,
    PUBLICATION_REMOTE_OBSERVATION_FIELDS,
    PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256,
    PUBLICATION_REMOTE_READBACK_EVIDENCE_FIELDS,
    PUBLICATION_REMOTE_REF_ABSENT_SENTINEL,
    PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
    PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL,
    PUBLICATION_WORKER_OWNERSHIP_FIELDS,
    build_contract_manifest,
    canonical_json_bytes,
    canonical_sha256,
)


EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-external-tag-message-v1"
)
EXTERNAL_PUBLICATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-external-publication-v1"
)
EXTERNAL_PUBLISHER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-git-tag-publisher-v1"
)
PUBLICATION_GENESIS_SHA256: Final[str] = "0" * 64
REMOTE_NAME: Final[str] = "origin"
ALLOWED_REMOTE_URL: Final[str] = (
    "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git"
)
FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE: Final[str] = (
    "!\"C:/Program Files/Git/mingw64/bin/git-credential-manager.exe\""
)
FROZEN_CREDENTIAL_HELPER_PATH: Final[str] = (
    "C:/Program Files/Git/mingw64/bin/git-credential-manager.exe"
)

ACQUISITION_PASS: Final[str] = "acquisition_pass"
SCORED_PASS: Final[str] = "scored_pass"
SCORED_FAILED_GATE: Final[str] = "scored_failed_gate"
FINAL_REGISTRY_SUCCESSOR: Final[str] = "final_registry_successor"
REPORT_KINDS: Final[tuple[str, ...]] = (
    ACQUISITION_PASS,
    SCORED_PASS,
    SCORED_FAILED_GATE,
    FINAL_REGISTRY_SUCCESSOR,
)

TERMINAL_PASS: Final[str] = "terminal_pass"
TERMINAL_FAIL: Final[str] = "terminal_fail"
REGISTERED_UNRUN: Final[str] = "registered_unrun"

NORMAL_PUBLICATION: Final[str] = "normal_publication"
PUBLICATION_RECOVERY: Final[str] = "publication_recovery"
OBSERVATION_OPERATION_KINDS: Final[tuple[str, ...]] = (
    NORMAL_PUBLICATION,
    PUBLICATION_RECOVERY,
)
PRE_PUSH: Final[str] = "pre_push"
POST_PUSH: Final[str] = "post_push"

READBACK_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "sec-gemma-online-risk-overlay-v2-2-"
    "publication-remote-readback-evidence-v1"
)
READBACK_EVIDENCE_VERIFIER_ID: Final[str] = (
    "sec-gemma-online-risk-overlay-v2-2-"
    "publication-remote-readback-evidence-verifier-v1"
)
READBACK_COMMAND_PROFILE_ID: Final[str] = (
    "sec-gemma-online-risk-overlay-v2-2-"
    "publication-remote-readback-command-profile-v1"
)
REMOTE_OBSERVATION_SCHEMA_VERSION: Final[str] = (
    "sec-gemma-online-risk-overlay-v2-2-publication-remote-observation-v1"
)
REMOTE_OBSERVATION_VERIFIER_ID: Final[str] = (
    "sec-gemma-online-risk-overlay-v2-2-"
    "publication-remote-observation-verifier-v1"
)

_REPORT_KIND_STATUS: Final[dict[str, str]] = {
    ACQUISITION_PASS: TERMINAL_PASS,
    SCORED_PASS: TERMINAL_PASS,
    SCORED_FAILED_GATE: TERMINAL_FAIL,
    FINAL_REGISTRY_SUCCESSOR: REGISTERED_UNRUN,
}
_SCORING_ATTEMPT_IDS: Final[frozenset[str]] = frozenset(
    {
        DEVELOPMENT_ATTEMPT_ID,
        CONFIRMATION_ATTEMPT_ID,
        FINAL_ATTEMPT_ID,
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_REMOTE_ROW_RE = re.compile(
    rb"(?P<oid>[0-9a-f]{40})\t(?P<ref>[^\r\n\t]+)\n"
)
_VERIFIED_EXTERNAL_PUBLICATION_SENTINEL = object()
_PREPARED_PUBLICATION_SENTINEL = object()
_PREPARED_TRANSPORT_SENTINEL = object()
_READBACK_RESULT_SENTINEL = object()
_PUSH_RESULT_SENTINEL = object()
_CONTAINED_EXECUTION_REQUEST_SENTINEL = object()

_HOST_ENVIRONMENT_KEYS: Final[tuple[str, ...]] = (
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATH",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "HOME",
)
_FIXED_CHILD_ENVIRONMENT: Final[dict[str, str]] = {
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "NUL",
    "GIT_CONFIG_GLOBAL": "NUL",
    "GIT_CONFIG_COUNT": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ALLOW_PROTOCOL": "https",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GCM_INTERACTIVE": "Never",
    "LANG": "C",
    "LC_ALL": "C",
}
_EXECUTABLE_PIN_FIELDS: Final[tuple[str, ...]] = (
    "git_executable_path",
    "git_executable_sha256",
    "git_version_stdout_sha256",
    "git_exec_path",
    "git_exec_path_directory_manifest_sha256",
    "git_remote_https_executable_path",
    "git_remote_https_executable_sha256",
    "credential_helper_config_value",
    "credential_helper_executable_path",
    "credential_helper_executable_sha256",
    "credential_helper_version_stdout_sha256",
    "command_interpreter_executable_path",
    "command_interpreter_executable_sha256",
    "transport_executable_closure_manifest_sha256",
)
_GIT_RUNTIME_CLOSURE_DIRECTORIES: Final[tuple[str, ...]] = (
    "cmd",
    "bin",
    "mingw64/bin",
    "mingw64/libexec/git-core",
    "usr/bin",
    "etc",
    "mingw64/etc",
)
_FINAL_REGISTRY_CONFIG_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r'\s*\[\s*([A-Za-z][A-Za-z0-9-]*)'
    r'\s*(?:"([^"\\\r\n]*)")?\s*\]'
    r'\s*(?:[#;].*)?\Z'
)
_FINAL_REGISTRY_CONFIG_KEY_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*([A-Za-z][A-Za-z0-9-]*)\s*(?:=(.*))?\Z"
)
_FINAL_REGISTRY_SAFE_CORE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "bare",
        "filemode",
        "ignorecase",
        "logallrefupdates",
        "precomposeunicode",
        "protecthfs",
        "protectntfs",
        "repositoryformatversion",
        "symlinks",
    }
)
_FINAL_REGISTRY_SAFE_EXTENSION_KEYS: Final[frozenset[str]] = frozenset(
    {"worktreeconfig"}
)
_FINAL_REGISTRY_SAFE_REMOTE_KEYS: Final[frozenset[str]] = frozenset(
    {"fetch", "url"}
)
_FINAL_REGISTRY_SAFE_BRANCH_KEYS: Final[frozenset[str]] = frozenset(
    {"merge", "remote", "vscode-merge-base"}
)


class SecGemmaOnlineRiskOverlayPublisherError(RuntimeError):
    """Publication material or transport differed from the frozen contract."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be one detached plain mapping"
        )
    return copy.deepcopy(value)


def _expect_fields(
    observed: Mapping[str, Any],
    fields: tuple[str, ...],
    location: str,
) -> None:
    if tuple(observed) != fields:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} fields differ from the frozen contract"
        )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _sha1(value: Any, location: str) -> str:
    if type(value) is not str or _SHA1_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be a lowercase Git SHA-1"
        )
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be an integer at least {minimum}"
        )
    return value


def _absolute_path(value: Any, location: str) -> Path:
    if type(value) is not str or not value:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be a non-empty absolute path string"
        )
    path = Path(value)
    if not path.is_absolute():
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"{location} must be absolute"
        )
    return path


def _portable_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    return digest.hexdigest()
                digest.update(block)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"Pinned executable is unavailable: {path}"
        ) from exc


def _is_reparse_or_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.stat(), "st_file_attributes", 0)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"Path identity is unavailable: {path}"
        ) from exc
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _path_identity_sha256(path: Path, *, require_directory: bool) -> str:
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"Path identity is unavailable: {path}"
        ) from exc
    if _is_reparse_or_link(resolved):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"Path identity cannot use a link or reparse point: {path}"
        )
    if require_directory != resolved.is_dir():
        kind = "directory" if require_directory else "file"
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"Path identity is not the required {kind}: {path}"
        )
    material = {
        "resolved_path": _portable_path(resolved),
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "st_mode": int(info.st_mode),
        "is_directory": require_directory,
    }
    return canonical_sha256(material)


def _final_registry_config_key_is_safe(
    *,
    section: str,
    subsection: str | None,
    key: str,
) -> bool:
    """Allow only inert repository metadata needed by this exact checkout."""

    if section == "core" and subsection is None:
        return key in _FINAL_REGISTRY_SAFE_CORE_KEYS
    if section == "extensions" and subsection is None:
        return key in _FINAL_REGISTRY_SAFE_EXTENSION_KEYS
    if (
        section == "remote"
        and subsection is not None
        and subsection.casefold() == REMOTE_NAME
    ):
        return key in _FINAL_REGISTRY_SAFE_REMOTE_KEYS
    if section == "branch" and subsection:
        return key in _FINAL_REGISTRY_SAFE_BRANCH_KEYS
    return False


def _parse_final_registry_config_entries(
    raw: bytes,
) -> list[tuple[str, str | None, str, str | None]]:
    """Parse only one deliberately strict, non-continuation Git syntax."""

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Final-registry local/worktree Git configuration is unsafe"
        ) from exc
    if "\x00" in text:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Final-registry local/worktree Git configuration is unsafe"
        )
    section: str | None = None
    subsection: str | None = None
    entries: list[tuple[str, str | None, str, str | None]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        trailing_backslashes = len(line) - len(line.rstrip("\\"))
        if trailing_backslashes % 2:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            )
        if stripped.startswith("["):
            matched_section = _FINAL_REGISTRY_CONFIG_SECTION_RE.fullmatch(
                line
            )
            if matched_section is None:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Final-registry local/worktree Git configuration is unsafe"
                )
            section = matched_section.group(1).casefold()
            subsection = matched_section.group(2)
            continue
        matched_key = _FINAL_REGISTRY_CONFIG_KEY_RE.fullmatch(line)
        if matched_key is None or section is None:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            )
        raw_value = matched_key.group(2)
        entries.append(
            (
                section,
                subsection,
                matched_key.group(1).casefold(),
                raw_value.strip() if raw_value is not None else None,
            )
        )
    return entries


def _test_only_final_registry_fixture_field(
    *,
    section: str,
    subsection: str | None,
    key: str,
    value: str | None,
    expected_rewrite_uri: str,
) -> str | None:
    exact_fields = {
        ("core", None, "autocrlf", "false"): "core.autocrlf",
        ("core", None, "longpaths", "true"): "core.longpaths",
        ("user", None, "name", "Registry Test"): "user.name",
        (
            "user",
            None,
            "email",
            "registry@example.invalid",
        ): "user.email",
        (
            "url",
            expected_rewrite_uri,
            "insteadof",
            ALLOWED_REMOTE_URL,
        ): "url.rewrite",
    }
    return exact_fields.get((section, subsection, key, value))


def _validate_final_registry_config_bytes(
    raw: bytes,
    *,
    location: str,
    test_only_rewrite_uri: str | None,
) -> None:
    """Reject every local Git setting outside one deliberately small allowlist.

    A strict parser is intentional here.  In particular, continuation lines,
    legacy dotted section syntax, includes, credential helpers, URL rewrites,
    HTTP options, filters, aliases, pagers, editors, hooks, fsmonitor, SSH
    commands, and every other command-bearing or transport-affecting setting
    are outside the allowlist and therefore fail closed.
    """

    observed_test_fields: set[str] = set()
    for section, subsection, key, value in (
        _parse_final_registry_config_entries(raw)
    ):
        if not _final_registry_config_key_is_safe(
            section=section,
            subsection=subsection,
            key=key,
        ):
            test_field = None
            if test_only_rewrite_uri is not None and location == "config":
                test_field = _test_only_final_registry_fixture_field(
                    section=section,
                    subsection=subsection,
                    key=key,
                    value=value,
                    expected_rewrite_uri=test_only_rewrite_uri,
                )
            if test_field is not None and test_field not in observed_test_fields:
                observed_test_fields.add(test_field)
                continue
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            )
    if test_only_rewrite_uri is not None and location == "config" and (
        observed_test_fields
        != {
            "core.autocrlf",
            "core.longpaths",
            "user.name",
            "user.email",
            "url.rewrite",
        }
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Final-registry local/worktree Git configuration is unsafe"
        )


def _final_registry_config_snapshot_sha256(
    git_directory: Path,
    *,
    test_only_rewrite_uri: str | None,
) -> str:
    """Audit and bind the exact local and optional worktree config bytes."""

    directory_identity = _path_identity_sha256(
        git_directory,
        require_directory=True,
    )
    rows: list[dict[str, Any]] = []
    for name, required in (("config", True), ("config.worktree", False)):
        path = git_directory / name
        if path.is_symlink():
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            )
        try:
            exists = path.exists()
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            ) from exc
        if not exists:
            if required:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Final-registry local/worktree Git configuration is unsafe"
                )
            rows.append({"name": name, "status": "absent"})
            continue
        if not path.is_file() or _is_reparse_or_link(path):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            )
        identity_before = _path_identity_sha256(
            path,
            require_directory=False,
        )
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            ) from exc
        identity_after = _path_identity_sha256(
            path,
            require_directory=False,
        )
        try:
            raw_after = path.read_bytes()
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            ) from exc
        if identity_before != identity_after or raw != raw_after:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration is unsafe"
            )
        _validate_final_registry_config_bytes(
            raw,
            location=name,
            test_only_rewrite_uri=(
                test_only_rewrite_uri if name == "config" else None
            ),
        )
        rows.append(
            {
                "name": name,
                "status": "present",
                "path_identity_sha256": identity_after,
                "byte_count": len(raw),
                "bytes_sha256": _bytes_sha256(raw),
            }
        )
    return canonical_sha256(
        {
            "schema_version": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "final-registry-local-config-snapshot-v1"
            ),
            "git_directory_identity_sha256": directory_identity,
            "files": rows,
        }
    )


def _test_only_bare_repo_identity_sha256(path: Path) -> str:
    """Bind the exact local bare repository allowed by integration tests."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Test-only final-registry rewrite target is not one exact bare repo"
        ) from exc
    if (
        path != resolved
        or path.name != "remote.git"
        or not path.is_dir()
        or _is_reparse_or_link(path)
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Test-only final-registry rewrite target is not one exact bare repo"
        )
    required_directories = ("objects", "refs", "hooks")
    required_files = ("HEAD", "config")
    for name in required_directories:
        candidate = path / name
        if not candidate.is_dir() or _is_reparse_or_link(candidate):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Test-only final-registry rewrite target is not one exact bare repo"
            )
    file_rows: list[dict[str, Any]] = []
    raw_config: bytes | None = None
    for name in required_files:
        candidate = path / name
        if not candidate.is_file() or _is_reparse_or_link(candidate):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Test-only final-registry rewrite target is not one exact bare repo"
            )
        try:
            raw = candidate.read_bytes()
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Test-only final-registry rewrite target is not one exact bare repo"
            ) from exc
        if name == "config":
            raw_config = raw
        file_rows.append(
            {
                "name": name,
                "path_identity_sha256": _path_identity_sha256(
                    candidate,
                    require_directory=False,
                ),
                "bytes_sha256": _bytes_sha256(raw),
            }
        )
    assert raw_config is not None
    config_entries = _parse_final_registry_config_entries(raw_config)
    bare_values = [
        value
        for section, subsection, key, value in config_entries
        if section == "core" and subsection is None and key == "bare"
    ]
    for section, subsection, key, value in config_entries:
        if (
            section != "core"
            or subsection is not None
            or key
            not in (_FINAL_REGISTRY_SAFE_CORE_KEYS | {"longpaths"})
            or (key == "longpaths" and value != "true")
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Test-only final-registry rewrite target is not one exact bare repo"
            )
    if bare_values != ["true"]:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Test-only final-registry rewrite target is not one exact bare repo"
        )
    try:
        hook_entries = list((path / "hooks").iterdir())
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Test-only final-registry rewrite target is not one exact bare repo"
        ) from exc
    if any(
        not entry.is_file()
        or _is_reparse_or_link(entry)
        or entry.suffix.casefold() != ".sample"
        for entry in hook_entries
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Test-only final-registry rewrite target is not one exact bare repo"
        )
    return canonical_sha256(
        {
            "schema_version": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "test-only-bare-repo-identity-v1"
            ),
            "root_identity_sha256": _path_identity_sha256(
                path,
                require_directory=True,
            ),
            "directory_identities": {
                name: _path_identity_sha256(
                    path / name,
                    require_directory=True,
                )
                for name in required_directories
            },
            "hooks_manifest_sha256": _directory_manifest_sha256(
                path / "hooks"
            ),
            "files": file_rows,
        }
    )


def _directory_manifest_sha256(path: Path) -> str:
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"Executable directory is unavailable: {path}"
        ) from exc
    if not root.is_dir() or _is_reparse_or_link(root):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Executable directory must be one real directory"
        )
    rows: list[dict[str, Any]] = []
    try:
        children = sorted(
            root.rglob("*"),
            key=lambda item: _portable_path(item.relative_to(root)),
        )
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Executable directory could not be enumerated"
        ) from exc
    for child in children:
        if _is_reparse_or_link(child):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Executable directory manifest contains a link or reparse point"
            )
        relative = _portable_path(child.relative_to(root))
        if child.is_dir():
            rows.append({"path": relative, "kind": "directory"})
        elif child.is_file():
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": child.stat().st_size,
                    "sha256": _file_sha256(child),
                }
            )
        else:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Executable directory contains an unsupported entry"
            )
    return canonical_sha256(rows)


def git_runtime_dependency_closure_sha256(git_root: Path) -> str:
    """Hash the complete non-system executable/config closure used by Git."""

    if not isinstance(git_root, Path) or not git_root.is_absolute():
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Git runtime closure root must be one absolute pathlib.Path"
        )
    try:
        root = git_root.resolve(strict=True)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Git runtime closure root is unavailable"
        ) from exc
    if not root.is_dir() or _is_reparse_or_link(root):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Git runtime closure root must be one real directory"
        )
    rows: list[dict[str, str]] = []
    for relative in _GIT_RUNTIME_CLOSURE_DIRECTORIES:
        directory = root / Path(relative)
        rows.append(
            {
                "relative_directory": relative,
                "directory_identity_sha256": _path_identity_sha256(
                    directory,
                    require_directory=True,
                ),
                "directory_manifest_sha256": _directory_manifest_sha256(
                    directory
                ),
            }
        )
    return canonical_sha256(
        {
            "schema_version": (
                "sec-gemma-online-risk-overlay-v2-2-"
                "git-runtime-dependency-closure-v1"
            ),
            "git_root_identity_sha256": _path_identity_sha256(
                root,
                require_directory=True,
            ),
            "directories": rows,
        }
    )


def _git_runtime_root_from_pins(pins: Mapping[str, Any]) -> Path:
    credential_helper = Path(
        pins["credential_helper_executable_path"]
    ).resolve(strict=True)
    try:
        root = credential_helper.parents[2]
    except IndexError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Pinned credential helper has no Git installation root"
        ) from exc
    expected = {
        "git_executable_path": root / "mingw64" / "bin" / "git.exe",
        "git_exec_path": root / "mingw64" / "libexec" / "git-core",
        "git_remote_https_executable_path": (
            root
            / "mingw64"
            / "libexec"
            / "git-core"
            / "git-remote-https.exe"
        ),
        "credential_helper_executable_path": (
            root / "mingw64" / "bin" / "git-credential-manager.exe"
        ),
        "command_interpreter_executable_path": (
            root / "usr" / "bin" / "sh.exe"
        ),
    }
    for field, path in expected.items():
        try:
            observed = Path(pins[field]).resolve(strict=True)
            required = path.resolve(strict=True)
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                f"Pinned Git runtime path is unavailable: {field}"
            ) from exc
        if observed != required:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                f"Pinned Git runtime path is not the real executable: {field}"
            )
    return root


def build_transport_executable_identity_pins(
    *,
    git_executable_path: Path,
    git_version_stdout: bytes,
    git_exec_path: Path,
    git_remote_https_executable_path: Path,
    credential_helper_executable_path: Path,
    credential_helper_version_stdout: bytes,
    command_interpreter_executable_path: Path,
) -> dict[str, Any]:
    """Hash the exact executable closure without launching a process."""

    for name, path in (
        ("git executable", git_executable_path),
        ("Git exec path", git_exec_path),
        ("git-remote-https executable", git_remote_https_executable_path),
        ("credential helper executable", credential_helper_executable_path),
        ("command interpreter executable", command_interpreter_executable_path),
    ):
        if not isinstance(path, Path) or not path.is_absolute():
            raise SecGemmaOnlineRiskOverlayPublisherError(
                f"{name} path must be one absolute pathlib.Path"
            )
    if (
        _portable_path(credential_helper_executable_path)
        != FROZEN_CREDENTIAL_HELPER_PATH
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Credential helper path differs from the frozen shell snippet"
        )
    if type(git_version_stdout) is not bytes or type(
        credential_helper_version_stdout
    ) is not bytes:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Executable version output must be exact raw bytes"
        )
    body = {
        "git_executable_path": _portable_path(git_executable_path),
        "git_executable_sha256": _file_sha256(git_executable_path),
        "git_version_stdout_sha256": _bytes_sha256(git_version_stdout),
        "git_exec_path": _portable_path(git_exec_path),
        "git_exec_path_directory_manifest_sha256": (
            _directory_manifest_sha256(git_exec_path)
        ),
        "git_remote_https_executable_path": _portable_path(
            git_remote_https_executable_path
        ),
        "git_remote_https_executable_sha256": _file_sha256(
            git_remote_https_executable_path
        ),
        "credential_helper_config_value": (
            FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE
        ),
        "credential_helper_executable_path": _portable_path(
            credential_helper_executable_path
        ),
        "credential_helper_executable_sha256": _file_sha256(
            credential_helper_executable_path
        ),
        "credential_helper_version_stdout_sha256": _bytes_sha256(
            credential_helper_version_stdout
        ),
        "command_interpreter_executable_path": _portable_path(
            command_interpreter_executable_path
        ),
        "command_interpreter_executable_sha256": _file_sha256(
            command_interpreter_executable_path
        ),
    }
    git_root = credential_helper_executable_path.resolve(strict=True).parents[2]
    return {
        **body,
        "transport_executable_closure_manifest_sha256": (
            git_runtime_dependency_closure_sha256(git_root)
        ),
    }


def validate_transport_executable_identity_pins(
    value: Any,
) -> dict[str, Any]:
    pins = _mapping(value, "transport executable identity pins")
    _expect_fields(pins, _EXECUTABLE_PIN_FIELDS, "transport executable pins")
    for field in (
        "git_executable_path",
        "git_exec_path",
        "git_remote_https_executable_path",
        "credential_helper_executable_path",
        "command_interpreter_executable_path",
    ):
        _absolute_path(pins[field], f"transport pin {field}")
    for field in (
        "git_executable_sha256",
        "git_version_stdout_sha256",
        "git_exec_path_directory_manifest_sha256",
        "git_remote_https_executable_sha256",
        "credential_helper_executable_sha256",
        "credential_helper_version_stdout_sha256",
        "command_interpreter_executable_sha256",
        "transport_executable_closure_manifest_sha256",
    ):
        _sha256(pins[field], f"transport pin {field}")
    if (
        pins["credential_helper_config_value"]
        != FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE
        or _portable_path(Path(pins["credential_helper_executable_path"]))
        != FROZEN_CREDENTIAL_HELPER_PATH
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Credential helper identity differs from the frozen contract"
        )
    return pins


def verify_pinned_transport_executables(
    pins: Mapping[str, Any],
    *,
    expected_dependency_closure_sha256: str | None = None,
) -> None:
    """Re-hash every pinned executable identity immediately before spawn."""

    validated = validate_transport_executable_identity_pins(pins)
    checks = (
        ("git_executable_path", "git_executable_sha256"),
        (
            "git_remote_https_executable_path",
            "git_remote_https_executable_sha256",
        ),
        (
            "credential_helper_executable_path",
            "credential_helper_executable_sha256",
        ),
        (
            "command_interpreter_executable_path",
            "command_interpreter_executable_sha256",
        ),
    )
    for path_field, hash_field in checks:
        if not hmac.compare_digest(
            _file_sha256(Path(validated[path_field])),
            validated[hash_field],
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                f"Pinned executable changed: {path_field}"
            )
    if not hmac.compare_digest(
        _directory_manifest_sha256(Path(validated["git_exec_path"])),
        validated["git_exec_path_directory_manifest_sha256"],
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Pinned Git exec-path directory changed"
        )
    expected = _sha256(
        validated["transport_executable_closure_manifest_sha256"],
        "durable Git runtime dependency closure",
    )
    if expected_dependency_closure_sha256 is not None and not (
        hmac.compare_digest(
            _sha256(
                expected_dependency_closure_sha256,
                "Git runtime dependency closure",
            ),
            expected,
        )
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Runtime and durable Git dependency closures differ"
        )
    root = _git_runtime_root_from_pins(validated)
    if not hmac.compare_digest(
        git_runtime_dependency_closure_sha256(root),
        expected,
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Pinned Git runtime dependency closure changed"
        )


def build_exact_child_environment(
    *,
    host_values: Mapping[str, str],
    executable_pins: Mapping[str, Any],
) -> dict[str, str]:
    """Construct a new exact environment without inheriting the parent."""

    if type(host_values) is not dict or tuple(host_values) != _HOST_ENVIRONMENT_KEYS:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Host child-environment values must use the exact frozen key order"
        )
    if len({name.casefold() for name in host_values}) != len(host_values):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Child environment contains case-fold duplicate names"
        )
    for name, value in host_values.items():
        if type(value) is not str or not value:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                f"Child environment {name} must be one non-empty string"
            )
    pins = validate_transport_executable_identity_pins(executable_pins)
    environment: dict[str, str] = dict(host_values)
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "NUL",
            "GIT_CONFIG_GLOBAL": "NUL",
            "GIT_CONFIG_COUNT": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": "https",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_EXEC_PATH": pins["git_exec_path"],
            "GCM_INTERACTIVE": "Never",
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    expected_keys = tuple(
        build_contract_manifest()["execution_integrity"][
            "publication_transport_isolation"
        ]["child_environment_key_set"]
    )
    if tuple(environment) != expected_keys:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Child environment key set differs from the frozen contract"
        )
    return environment


def _attempt_and_status(
    *,
    attempt_id: Any,
    terminal_status: Any,
    report_kind: Any,
) -> tuple[str, str, str]:
    if type(report_kind) is not str or report_kind not in REPORT_KINDS:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication report kind is not preregistered"
        )
    expected_status = _REPORT_KIND_STATUS[report_kind]
    if terminal_status != expected_status:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication terminal status differs from its report kind"
        )
    if report_kind == ACQUISITION_PASS:
        expected_attempts = {DEVELOPMENT_ACQUISITION_ID}
    elif report_kind == FINAL_REGISTRY_SUCCESSOR:
        expected_attempts = {FINAL_ATTEMPT_ID}
    else:
        expected_attempts = _SCORING_ATTEMPT_IDS
    if type(attempt_id) is not str or attempt_id not in expected_attempts:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication attempt differs from its report kind"
        )
    return attempt_id, terminal_status, report_kind


def _tag_ref(*, attempt_id: str, report_kind: str) -> str:
    template = (
        FINAL_REGISTRY_TAG_REF_TEMPLATE
        if report_kind == FINAL_REGISTRY_SUCCESSOR
        else EXTERNAL_TAG_REF_TEMPLATE
    )
    return template.format(attempt_id=attempt_id)


def build_external_tag_message(
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_id: str,
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
    predecessor_publication_sha256: str,
) -> dict[str, Any]:
    """Build the exact canonical annotated-tag message."""

    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    fixed_attempt, fixed_status, fixed_kind = _attempt_and_status(
        attempt_id=attempt_id,
        terminal_status=terminal_status,
        report_kind=report_kind,
    )
    message = {
        "schema_version": EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_commit": implementation["implementation_commit"],
        "attempt_id": fixed_attempt,
        "terminal_status": fixed_status,
        "report_kind": fixed_kind,
        "artifact_sha256": _sha256(
            artifact_sha256,
            "publication artifact",
        ),
        "predecessor_publication_sha256": _sha256(
            predecessor_publication_sha256,
            "predecessor publication",
        ),
        "external_cost_usd": 0.0,
    }
    _expect_fields(message, EXTERNAL_TAG_MESSAGE_FIELDS, "external tag message")
    return message


def validate_external_tag_message(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _mapping(value, "external tag message")
    _expect_fields(
        observed,
        EXTERNAL_TAG_MESSAGE_FIELDS,
        "external tag message",
    )
    expected = build_external_tag_message(
        implementation_manifest=implementation_manifest,
        attempt_id=observed["attempt_id"],
        terminal_status=observed["terminal_status"],
        report_kind=observed["report_kind"],
        artifact_sha256=observed["artifact_sha256"],
        predecessor_publication_sha256=observed[
            "predecessor_publication_sha256"
        ],
    )
    if observed != expected:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External tag message differs from its canonical reconstruction"
        )
    return expected


def _publication_body(
    *,
    implementation_manifest: Mapping[str, Any],
    tag_message: Mapping[str, Any],
    tag_ref: str,
    remote_name: str,
    remote_url: str,
    remote_tag_object_sha1: str,
    remote_peeled_commit: str,
) -> dict[str, Any]:
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    message = validate_external_tag_message(
        tag_message,
        implementation_manifest=implementation,
    )
    if tag_ref != _tag_ref(
        attempt_id=message["attempt_id"],
        report_kind=message["report_kind"],
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication tag ref changed"
        )
    if remote_name != REMOTE_NAME:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication must bind the frozen origin name"
        )
    if (
        remote_url != ALLOWED_REMOTE_URL
        or remote_url != implementation["origin_url"]
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication remote URL changed"
        )
    tag_object = _sha1(
        remote_tag_object_sha1,
        "remote annotated-tag object",
    )
    peeled = _sha1(remote_peeled_commit, "remote peeled commit")
    if peeled != implementation["implementation_commit"]:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Remote tag does not peel to the implementation commit"
        )
    return {
        "schema_version": EXTERNAL_PUBLICATION_SCHEMA_VERSION,
        "publisher_id": EXTERNAL_PUBLISHER_ID,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_commit": implementation["implementation_commit"],
        "attempt_id": message["attempt_id"],
        "terminal_status": message["terminal_status"],
        "report_kind": message["report_kind"],
        "artifact_sha256": message["artifact_sha256"],
        "predecessor_publication_sha256": message[
            "predecessor_publication_sha256"
        ],
        "tag_ref": tag_ref,
        "tag_target_commit": implementation["implementation_commit"],
        "tag_message_sha256": canonical_sha256(message),
        "remote_name": remote_name,
        "remote_url": remote_url,
        "remote_tag_object_sha1": tag_object,
        "remote_peeled_commit": peeled,
        "external_cost_usd": 0.0,
    }


def build_external_publication(
    *,
    implementation_manifest: Mapping[str, Any],
    tag_message: Mapping[str, Any],
    tag_ref: str,
    remote_name: str,
    remote_url: str,
    remote_tag_object_sha1: str,
    remote_peeled_commit: str,
) -> dict[str, Any]:
    body = _publication_body(
        implementation_manifest=implementation_manifest,
        tag_message=tag_message,
        tag_ref=tag_ref,
        remote_name=remote_name,
        remote_url=remote_url,
        remote_tag_object_sha1=remote_tag_object_sha1,
        remote_peeled_commit=remote_peeled_commit,
    )
    publication = {**body, "publication_sha256": canonical_sha256(body)}
    _expect_fields(
        publication,
        EXTERNAL_PUBLICATION_FIELDS,
        "external publication",
    )
    return publication


def validate_external_publication(
    value: Any,
    *,
    implementation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _mapping(value, "external publication")
    _expect_fields(
        observed,
        EXTERNAL_PUBLICATION_FIELDS,
        "external publication",
    )
    digest = _sha256(
        observed["publication_sha256"],
        "external publication self-hash",
    )
    message = build_external_tag_message(
        implementation_manifest=implementation_manifest,
        attempt_id=observed["attempt_id"],
        terminal_status=observed["terminal_status"],
        report_kind=observed["report_kind"],
        artifact_sha256=observed["artifact_sha256"],
        predecessor_publication_sha256=observed[
            "predecessor_publication_sha256"
        ],
    )
    expected = build_external_publication(
        implementation_manifest=implementation_manifest,
        tag_message=message,
        tag_ref=observed["tag_ref"],
        remote_name=observed["remote_name"],
        remote_url=observed["remote_url"],
        remote_tag_object_sha1=observed["remote_tag_object_sha1"],
        remote_peeled_commit=observed["remote_peeled_commit"],
    )
    if (
        not hmac.compare_digest(digest, expected["publication_sha256"])
        or observed != expected
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "External publication differs from its canonical reconstruction"
        )
    return expected


def _git_object_sha1(object_type: str, payload: bytes) -> str:
    material = (
        f"{object_type} {len(payload)}\0".encode("ascii") + payload
    )
    try:
        digest = hashlib.sha1(material, usedforsecurity=False)
    except TypeError:
        digest = hashlib.sha1(material)
    return digest.hexdigest()


def _canonical_tag_object_bytes(
    *,
    implementation_commit: str,
    tag_ref: str,
    message: Mapping[str, Any],
) -> bytes:
    tag_name = tag_ref.removeprefix("refs/tags/")
    if tag_name == tag_ref or not tag_name:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Canonical tag ref is not under refs/tags"
        )
    header = (
        f"object {implementation_commit}\n"
        "type commit\n"
        f"tag {tag_name}\n"
        "tagger SEC Gemma Online Risk Overlay Publisher "
        "<sec-gemma-online-risk-overlay@localhost> 0 +0000\n\n"
    ).encode("utf-8")
    return header + canonical_json_bytes(message) + b"\n"


class PreparedExternalPublication:
    """Opaque deterministic material that can be rebuilt after a restart."""

    __slots__ = (
        "_implementation",
        "_tag_message",
        "_tag_ref",
        "_tag_object_bytes",
        "_tag_object_sha1",
        "_expected_publication",
        "_sentinel",
    )

    def __init__(
        self,
        *,
        implementation: Mapping[str, Any],
        tag_message: Mapping[str, Any],
        tag_ref: str,
        tag_object_bytes: bytes,
        tag_object_sha1: str,
        expected_publication: Mapping[str, Any],
        _sentinel: object,
    ) -> None:
        if _sentinel is not _PREPARED_PUBLICATION_SENTINEL:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Prepared publications can only be issued by this module"
            )
        self._implementation = copy.deepcopy(dict(implementation))
        self._tag_message = copy.deepcopy(dict(tag_message))
        self._tag_ref = tag_ref
        self._tag_object_bytes = bytes(tag_object_bytes)
        self._tag_object_sha1 = tag_object_sha1
        self._expected_publication = copy.deepcopy(
            dict(expected_publication)
        )
        self._sentinel = _sentinel

    @property
    def implementation_manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._implementation)

    @property
    def tag_message(self) -> dict[str, Any]:
        return copy.deepcopy(self._tag_message)

    @property
    def tag_message_bytes(self) -> bytes:
        return canonical_json_bytes(self._tag_message)

    @property
    def tag_ref(self) -> str:
        return self._tag_ref

    @property
    def tag_object_bytes(self) -> bytes:
        return bytes(self._tag_object_bytes)

    @property
    def expected_tag_object_sha1(self) -> str:
        return self._tag_object_sha1

    @property
    def expected_peeled_commit(self) -> str:
        return self._implementation["implementation_commit"]

    @property
    def expected_publication(self) -> dict[str, Any]:
        return copy.deepcopy(self._expected_publication)

    @property
    def intent_material(self) -> dict[str, Any]:
        return {
            "tag_ref": self._tag_ref,
            "tag_target_commit": self.expected_peeled_commit,
            "tag_message_sha256": canonical_sha256(self._tag_message),
            "expected_tag_object_sha1": self._tag_object_sha1,
            "expected_peeled_commit": self.expected_peeled_commit,
            "expected_publication_sha256": self._expected_publication[
                "publication_sha256"
            ],
            "remote_name": REMOTE_NAME,
            "remote_url": ALLOWED_REMOTE_URL,
        }

    def __repr__(self) -> str:
        return "PreparedExternalPublication(<redacted>)"


def is_prepared_external_publication(value: Any) -> bool:
    return (
        type(value) is PreparedExternalPublication
        and getattr(value, "_sentinel", None)
        is _PREPARED_PUBLICATION_SENTINEL
    )


def prepare_external_publication(
    *,
    implementation_manifest: Mapping[str, Any],
    attempt_id: str,
    terminal_status: str,
    report_kind: str,
    artifact_sha256: str,
    predecessor_publication_sha256: str,
) -> PreparedExternalPublication:
    implementation = validate_implementation_manifest(
        implementation_manifest
    )
    message = build_external_tag_message(
        implementation_manifest=implementation,
        attempt_id=attempt_id,
        terminal_status=terminal_status,
        report_kind=report_kind,
        artifact_sha256=artifact_sha256,
        predecessor_publication_sha256=predecessor_publication_sha256,
    )
    tag_ref = _tag_ref(
        attempt_id=message["attempt_id"],
        report_kind=message["report_kind"],
    )
    tag_object_bytes = _canonical_tag_object_bytes(
        implementation_commit=implementation["implementation_commit"],
        tag_ref=tag_ref,
        message=message,
    )
    tag_object_sha1 = _git_object_sha1("tag", tag_object_bytes)
    expected_publication = build_external_publication(
        implementation_manifest=implementation,
        tag_message=message,
        tag_ref=tag_ref,
        remote_name=REMOTE_NAME,
        remote_url=ALLOWED_REMOTE_URL,
        remote_tag_object_sha1=tag_object_sha1,
        remote_peeled_commit=implementation["implementation_commit"],
    )
    return PreparedExternalPublication(
        implementation=implementation,
        tag_message=message,
        tag_ref=tag_ref,
        tag_object_bytes=tag_object_bytes,
        tag_object_sha1=tag_object_sha1,
        expected_publication=expected_publication,
        _sentinel=_PREPARED_PUBLICATION_SENTINEL,
    )


def reconstruct_prepared_external_publication(
    *,
    implementation_manifest: Mapping[str, Any],
    publication_intent: Any,
    intent_validator: Callable[[Any], Mapping[str, Any]],
) -> PreparedExternalPublication:
    """Rebuild deterministic tag material from one opaque durable intent."""

    if not callable(intent_validator):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Publication intent validator must be callable"
        )
    observed = _mapping(
        intent_validator(publication_intent),
        "validated publication intent",
    )
    _expect_fields(observed, PUBLICATION_INTENT_FIELDS, "publication intent")
    _sha256(observed["publication_intent_sha256"], "publication intent hash")
    if canonical_sha256(
        {
            key: value
            for key, value in observed.items()
            if key != "publication_intent_sha256"
        }
    ) != observed["publication_intent_sha256"]:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Publication intent self-hash is inconsistent"
        )
    prepared = prepare_external_publication(
        implementation_manifest=implementation_manifest,
        attempt_id=observed["attempt_id"],
        terminal_status=observed["terminal_status"],
        report_kind=observed["report_kind"],
        artifact_sha256=observed["artifact_sha256"],
        predecessor_publication_sha256=observed[
            "predecessor_publication_sha256"
        ],
    )
    for field, expected in prepared.intent_material.items():
        if observed[field] != expected:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                f"Publication intent changed deterministic field {field}"
            )
    return prepared


def _config_bytes(*, hooks_path: Path) -> tuple[bytes, list[list[str]]]:
    hooks = _portable_path(hooks_path)
    entries = [
        ["core.repositoryformatversion", "0"],
        ["core.bare", "true"],
        ["credential.helper", FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE],
        ["core.hooksPath", hooks],
        ["credential.useHttpPath", "true"],
        ["http.followRedirects", "false"],
        ["http.sslVerify", "true"],
    ]
    text = (
        "[core]\n"
        "\trepositoryformatversion = 0\n"
        "\tbare = true\n"
        f'\thooksPath = "{hooks}"\n'
        "[credential]\n"
        f"\thelper = {FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE}\n"
        "\tuseHttpPath = true\n"
        "[http]\n"
        "\tfollowRedirects = false\n"
        "\tsslVerify = true\n"
    )
    return text.encode("utf-8"), entries


def _write_exact(path: Path, payload: bytes) -> None:
    try:
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    f"Existing isolated transport file differs: {path}"
                )
            return
        path.write_bytes(payload)
    except SecGemmaOnlineRiskOverlayPublisherError:
        raise
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            f"Isolated transport file could not be written: {path}"
        ) from exc


def _loose_object_bytes(payload: bytes) -> bytes:
    header = f"tag {len(payload)}\0".encode("ascii")
    return zlib.compress(header + payload)


def _transport_entry_manifest(
    *,
    directory: Path,
    tag_object_path: Path,
) -> list[dict[str, str]]:
    expected_paths = {
        "HEAD": "file",
        "config": "file",
        "hooks": "directory",
        "objects": "directory",
        "objects/info": "directory",
        "objects/info/alternates": "file",
        "objects/pack": "directory",
        _portable_path(tag_object_path.parent.relative_to(directory)): (
            "directory"
        ),
        _portable_path(tag_object_path.relative_to(directory)): "file",
        "refs": "directory",
        "refs/heads": "directory",
        "refs/tags": "directory",
    }
    try:
        observed_paths = {
            _portable_path(path.relative_to(directory)): (
                "directory"
                if path.is_dir()
                else "file"
                if path.is_file()
                else "unsupported"
            )
            for path in directory.rglob("*")
        }
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated transport directory could not be enumerated"
        ) from exc
    if observed_paths != expected_paths:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated transport directory contains a missing or extra entry"
        )
    return [
        {"path": path, "kind": expected_paths[path]}
        for path in sorted(expected_paths)
    ]


def _validate_operation(
    *,
    operation_kind: Any,
    operation_sha256: Any,
) -> tuple[str, str]:
    if operation_kind not in OBSERVATION_OPERATION_KINDS:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Publication operation kind is not frozen"
        )
    operation_hash = _sha256(
        operation_sha256,
        "publication operation hash",
    )
    if (
        operation_kind == NORMAL_PUBLICATION
        and operation_hash != PUBLICATION_NORMAL_OPERATION_SHA256
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Normal publication operation hash changed"
        )
    return operation_kind, operation_hash


def _transport_contract() -> dict[str, Any]:
    return build_contract_manifest()["execution_integrity"][
        "publication_transport_isolation"
    ]


def _readback_contract_profile() -> dict[str, Any]:
    return build_contract_manifest()["execution_integrity"][
        "publication_remote_observation"
    ]["readback_command_profile"]


def _push_contract_profile() -> dict[str, Any]:
    return build_contract_manifest()["execution_integrity"][
        "publication_recovery"
    ]["pre_push_authorization_record"]["push_command_profile"]


TRANSPORT_ISOLATION_PROFILE_SHA256: Final[str] = canonical_sha256(
    _transport_contract()
)


class PreparedPublicationTransport:
    """Opaque exact isolated Git directory and its pinned execution profile."""

    __slots__ = (
        "_manifest",
        "_directory",
        "_source_objects",
        "_config_bytes",
        "_alternates_bytes",
        "_tag_loose_bytes",
        "_tag_object_path",
        "_environment",
        "_pins",
        "_identity_verifier",
        "_sentinel",
    )

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        directory: Path,
        source_objects: Path,
        config_bytes: bytes,
        alternates_bytes: bytes,
        tag_loose_bytes: bytes,
        tag_object_path: Path,
        environment: Mapping[str, str],
        pins: Mapping[str, Any],
        identity_verifier: Callable[[Mapping[str, Any]], None],
        _sentinel: object,
    ) -> None:
        if _sentinel is not _PREPARED_TRANSPORT_SENTINEL:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Prepared transports can only be issued by this module"
            )
        self._manifest = copy.deepcopy(dict(manifest))
        self._directory = directory
        self._source_objects = source_objects
        self._config_bytes = bytes(config_bytes)
        self._alternates_bytes = bytes(alternates_bytes)
        self._tag_loose_bytes = bytes(tag_loose_bytes)
        self._tag_object_path = tag_object_path
        self._environment = dict(environment)
        self._pins = copy.deepcopy(dict(pins))
        self._identity_verifier = identity_verifier
        self._sentinel = _sentinel

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    @property
    def manifest_sha256(self) -> str:
        return self._manifest[
            "isolated_transport_git_directory_manifest_sha256"
        ]

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def environment(self) -> dict[str, str]:
        return dict(self._environment)

    @property
    def executable_pins(self) -> dict[str, Any]:
        return copy.deepcopy(self._pins)

    def __repr__(self) -> str:
        return "PreparedPublicationTransport(<redacted>)"


def is_prepared_publication_transport(value: Any) -> bool:
    return (
        type(value) is PreparedPublicationTransport
        and getattr(value, "_sentinel", None)
        is _PREPARED_TRANSPORT_SENTINEL
    )


def _validate_transport_manifest(value: Any) -> dict[str, Any]:
    observed = _mapping(value, "isolated transport manifest")
    fields = tuple(_transport_contract()["isolated_git_directory_manifest_fields"])
    _expect_fields(observed, fields, "isolated transport manifest")
    digest = _sha256(
        observed["isolated_transport_git_directory_manifest_sha256"],
        "isolated transport manifest self-hash",
    )
    body = {
        key: item
        for key, item in observed.items()
        if key != "isolated_transport_git_directory_manifest_sha256"
    }
    if not hmac.compare_digest(digest, canonical_sha256(body)):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated transport manifest self-hash is inconsistent"
        )
    if (
        observed["schema_version"]
        != _transport_contract()["schema_version"]
        or observed["profile_id"] != _transport_contract()["profile_id"]
        or observed["contract_version"] != CONTRACT_VERSION
        or observed["contract_sha256"] != CONTRACT_SHA256
        or observed["path_lookup_forbidden"] is not True
        or observed["remote_url"] != ALLOWED_REMOTE_URL
        or observed["remote_url_scheme"] != "https"
        or observed["readback_command_profile_sha256"]
        != PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256
        or observed["push_command_profile_sha256"]
        != PUBLICATION_PUSH_COMMAND_PROFILE_SHA256
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated transport manifest fixed values changed"
        )
    return observed


def prepare_isolated_publication_transport(
    *,
    transport_root: Path,
    source_object_directory: Path,
    prepared_publication: PreparedExternalPublication,
    executable_pins: Mapping[str, Any],
    host_environment_values: Mapping[str, str],
    identity_verifier: Callable[[Mapping[str, Any]], None],
    store_instance_id: str,
    store_session_nonce_sha256: str,
    publication_intent_sha256: str,
    operation_kind: str,
    operation_sha256: str,
    pre_transport_store_journal_sequence: int,
    pre_transport_store_journal_tip_sha256: str,
) -> PreparedPublicationTransport:
    """Create or exactly reopen the fresh config-isolated bare directory."""

    if not isinstance(transport_root, Path) or not transport_root.is_absolute():
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Transport root must be one absolute pathlib.Path"
        )
    if not isinstance(source_object_directory, Path) or not (
        source_object_directory.is_absolute()
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Source object directory must be one absolute pathlib.Path"
        )
    if not is_prepared_external_publication(prepared_publication):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Transport requires an opaque prepared publication"
        )
    if not callable(identity_verifier):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Transport executable identity verifier must be callable"
        )
    pins = validate_transport_executable_identity_pins(executable_pins)
    identity_verifier(copy.deepcopy(pins))
    environment = build_exact_child_environment(
        host_values=host_environment_values,
        executable_pins=pins,
    )
    fixed_operation_kind, fixed_operation_sha = _validate_operation(
        operation_kind=operation_kind,
        operation_sha256=operation_sha256,
    )
    store_id = _sha256(store_instance_id, "store instance ID")
    session_nonce = _sha256(
        store_session_nonce_sha256,
        "store session nonce",
    )
    intent_hash = _sha256(
        publication_intent_sha256,
        "publication intent",
    )
    journal_sequence = _integer(
        pre_transport_store_journal_sequence,
        "pre-transport journal sequence",
    )
    journal_tip = _sha256(
        pre_transport_store_journal_tip_sha256,
        "pre-transport journal tip",
    )
    implementation = prepared_publication.implementation_manifest
    try:
        source_objects = source_object_directory.resolve(strict=True)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Source object directory is unavailable"
        ) from exc
    source_identity = _path_identity_sha256(
        source_objects,
        require_directory=True,
    )
    try:
        transport_root.mkdir(parents=False, exist_ok=True)
        directory = transport_root / "transport.git"
        directory.mkdir(exist_ok=True)
        for relative in (
            "hooks",
            "objects",
            "objects/info",
            "objects/pack",
            "refs",
            "refs/heads",
            "refs/tags",
        ):
            (directory / relative).mkdir(exist_ok=True)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated transport directory could not be created"
        ) from exc
    hooks_path = directory / "hooks"
    config_bytes, config_entries = _config_bytes(hooks_path=hooks_path)
    alternates_bytes = (
        _portable_path(source_objects).encode("utf-8") + b"\n"
    )
    _write_exact(directory / "config", config_bytes)
    _write_exact(directory / "HEAD", b"ref: refs/heads/main\n")
    _write_exact(
        directory / "objects" / "info" / "alternates",
        alternates_bytes,
    )
    object_sha1 = prepared_publication.expected_tag_object_sha1
    object_directory = directory / "objects" / object_sha1[:2]
    try:
        object_directory.mkdir(exist_ok=True)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated tag object directory could not be created"
        ) from exc
    tag_object_path = object_directory / object_sha1[2:]
    tag_loose_bytes = _loose_object_bytes(
        prepared_publication.tag_object_bytes
    )
    _write_exact(tag_object_path, tag_loose_bytes)
    try:
        hook_entries = list(hooks_path.iterdir())
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated hooks directory could not be enumerated"
        ) from exc
    if hook_entries:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated hooks directory is not empty"
        )
    body = {
        "schema_version": _transport_contract()["schema_version"],
        "profile_id": _transport_contract()["profile_id"],
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "implementation_manifest_sha256": implementation[
            "implementation_manifest_sha256"
        ],
        "implementation_commit": implementation["implementation_commit"],
        "store_instance_id": store_id,
        "store_session_nonce_sha256": session_nonce,
        "attempt_id": prepared_publication.tag_message["attempt_id"],
        "publication_intent_sha256": intent_hash,
        "operation_kind": fixed_operation_kind,
        "operation_sha256": fixed_operation_sha,
        "isolated_git_directory_identity_sha256": _path_identity_sha256(
            directory,
            require_directory=True,
        ),
        "local_config_entries_sha256": canonical_sha256(config_entries),
        "empty_hooks_directory_path": _portable_path(hooks_path),
        "empty_hooks_directory_identity_sha256": _path_identity_sha256(
            hooks_path,
            require_directory=True,
        ),
        "empty_hooks_directory_listing_sha256": canonical_sha256([]),
        "alternates_file_bytes_sha256": _bytes_sha256(alternates_bytes),
        "alternate_object_directory_identity_sha256": source_identity,
        **pins,
        "child_environment_policy_id": _transport_contract()[
            "child_environment_policy_id"
        ],
        "child_environment_sha256": canonical_sha256(environment),
        "path_lookup_forbidden": True,
        "remote_url": ALLOWED_REMOTE_URL,
        "remote_url_scheme": "https",
        "readback_command_profile_sha256": (
            PUBLICATION_REMOTE_READBACK_COMMAND_PROFILE_SHA256
        ),
        "push_command_profile_sha256": (
            PUBLICATION_PUSH_COMMAND_PROFILE_SHA256
        ),
        "pre_transport_store_journal_sequence": journal_sequence,
        "pre_transport_store_journal_tip_sha256": journal_tip,
    }
    manifest = {
        **body,
        "isolated_transport_git_directory_manifest_sha256": (
            canonical_sha256(body)
        ),
    }
    _validate_transport_manifest(manifest)
    transport = PreparedPublicationTransport(
        manifest=manifest,
        directory=directory,
        source_objects=source_objects,
        config_bytes=config_bytes,
        alternates_bytes=alternates_bytes,
        tag_loose_bytes=tag_loose_bytes,
        tag_object_path=tag_object_path,
        environment=environment,
        pins=pins,
        identity_verifier=identity_verifier,
        _sentinel=_PREPARED_TRANSPORT_SENTINEL,
    )
    _verify_transport(transport, prepared_publication)
    return transport


def _verify_transport(
    transport: PreparedPublicationTransport,
    prepared_publication: PreparedExternalPublication,
) -> None:
    if not is_prepared_publication_transport(transport):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Remote command requires an opaque prepared transport"
        )
    if not is_prepared_external_publication(prepared_publication):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Remote command requires an opaque prepared publication"
        )
    manifest = _validate_transport_manifest(transport._manifest)
    implementation = prepared_publication.implementation_manifest
    if (
        manifest["implementation_manifest_sha256"]
        != implementation["implementation_manifest_sha256"]
        or manifest["implementation_commit"]
        != implementation["implementation_commit"]
        or manifest["attempt_id"]
        != prepared_publication.tag_message["attempt_id"]
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Transport crossed its implementation or attempt"
        )
    if _path_identity_sha256(
        transport._directory,
        require_directory=True,
    ) != manifest["isolated_git_directory_identity_sha256"]:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated transport directory identity changed"
        )
    hooks_path = Path(manifest["empty_hooks_directory_path"])
    if _path_identity_sha256(
        hooks_path,
        require_directory=True,
    ) != manifest["empty_hooks_directory_identity_sha256"]:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated hooks directory identity changed"
        )
    try:
        hook_entries = list(hooks_path.iterdir())
        config_bytes = (transport._directory / "config").read_bytes()
        alternates_bytes = (
            transport._directory / "objects" / "info" / "alternates"
        ).read_bytes()
        tag_loose_bytes = transport._tag_object_path.read_bytes()
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated transport files are unavailable"
        ) from exc
    expected_config_bytes, expected_config_entries = _config_bytes(
        hooks_path=hooks_path
    )
    _transport_entry_manifest(
        directory=transport._directory,
        tag_object_path=transport._tag_object_path,
    )
    if (
        hook_entries
        or canonical_sha256([]) != manifest[
            "empty_hooks_directory_listing_sha256"
        ]
        or canonical_sha256(expected_config_entries)
        != manifest["local_config_entries_sha256"]
        or expected_config_bytes != transport._config_bytes
        or config_bytes != transport._config_bytes
        or alternates_bytes != transport._alternates_bytes
        or tag_loose_bytes != transport._tag_loose_bytes
        or _bytes_sha256(alternates_bytes)
        != manifest["alternates_file_bytes_sha256"]
        or _path_identity_sha256(
            transport._source_objects,
            require_directory=True,
        )
        != manifest["alternate_object_directory_identity_sha256"]
        or canonical_sha256(transport._environment)
        != manifest["child_environment_sha256"]
        or any(
            manifest[field] != transport._pins[field]
            for field in _EXECUTABLE_PIN_FIELDS
        )
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Isolated transport filesystem or environment changed"
        )
    transport._identity_verifier(copy.deepcopy(transport._pins))


class PublicationProcessExecution:
    """Exact raw result returned by the runner's contained process executor."""

    __slots__ = ("process_exit_status", "process_exit_code", "stdout", "stderr")

    def __init__(
        self,
        *,
        process_exit_status: str,
        process_exit_code: int | None,
        stdout: bytes,
        stderr: bytes,
    ) -> None:
        if process_exit_status not in {
            "exited",
            "deadline",
            "parent_interrupted",
            "spawn_failed",
        }:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Process exit status is not frozen"
            )
        if process_exit_status == "exited":
            if type(process_exit_code) is not int:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Exited publication process requires an integer exit code"
                )
        elif process_exit_code is not None:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Non-exited publication process cannot have an exit code"
            )
        if type(stdout) is not bytes or type(stderr) is not bytes:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publication process output must remain exact raw bytes"
            )
        self.process_exit_status = process_exit_status
        self.process_exit_code = process_exit_code
        self.stdout = stdout
        self.stderr = stderr


class _ContainedPublicationExecutionRequest:
    """Opaque exact command request issued only after publisher validation."""

    __slots__ = (
        "_argv",
        "_cwd",
        "_environment",
        "_expected_tag_object_sha1",
        "_operation_kind",
        "_operation_sha256",
        "_pre_push_authorization_sha256",
        "_profile_kind",
        "_publication_intent_sha256",
        "_sentinel",
        "_tag_ref",
        "_timeout_seconds",
        "_transport_manifest_sha256",
        "_worker_ownership_sha256",
    )

    def __init__(
        self,
        *,
        argv: tuple[str, ...],
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: float,
        profile_kind: str,
        transport_manifest_sha256: str,
        publication_intent_sha256: str,
        operation_kind: str,
        operation_sha256: str,
        worker_ownership_sha256: str,
        pre_push_authorization_sha256: str | None,
        expected_tag_object_sha1: str,
        tag_ref: str,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _CONTAINED_EXECUTION_REQUEST_SENTINEL:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Contained execution requests can only be issued by the publisher"
            )
        self._argv = tuple(argv)
        self._cwd = cwd
        self._environment = dict(environment)
        self._timeout_seconds = float(timeout_seconds)
        self._profile_kind = profile_kind
        self._transport_manifest_sha256 = transport_manifest_sha256
        self._publication_intent_sha256 = publication_intent_sha256
        self._operation_kind = operation_kind
        self._operation_sha256 = operation_sha256
        self._worker_ownership_sha256 = worker_ownership_sha256
        self._pre_push_authorization_sha256 = (
            pre_push_authorization_sha256
        )
        self._expected_tag_object_sha1 = expected_tag_object_sha1
        self._tag_ref = tag_ref
        self._sentinel = _sentinel


def contained_publication_execution_request_material(
    request: Any,
) -> dict[str, Any]:
    """Validate and detach one opaque publisher-issued command request."""

    if (
        type(request) is not _ContainedPublicationExecutionRequest
        or request._sentinel is not _CONTAINED_EXECUTION_REQUEST_SENTINEL
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Contained process execution requires an opaque publisher request"
        )
    return {
        "argv": request._argv,
        "cwd": request._cwd,
        "environment": dict(request._environment),
        "timeout_seconds": request._timeout_seconds,
        "profile_kind": request._profile_kind,
        "transport_manifest_sha256": (
            request._transport_manifest_sha256
        ),
        "publication_intent_sha256": (
            request._publication_intent_sha256
        ),
        "operation_kind": request._operation_kind,
        "operation_sha256": request._operation_sha256,
        "worker_ownership_sha256": (
            request._worker_ownership_sha256
        ),
        "pre_push_authorization_sha256": (
            request._pre_push_authorization_sha256
        ),
        "expected_tag_object_sha1": (
            request._expected_tag_object_sha1
        ),
        "tag_ref": request._tag_ref,
    }


def _resolved_profile(
    *,
    template: Mapping[str, Any],
    transport: PreparedPublicationTransport,
    prepared: PreparedExternalPublication,
) -> dict[str, Any]:
    replacements = {
        "{verified_git_executable_path}": transport._pins[
            "git_executable_path"
        ],
        "verified_isolated_transport_git_directory": _portable_path(
            transport._directory
        ),
        "{remote_url}": ALLOWED_REMOTE_URL,
        "{tag_ref}": prepared.tag_ref,
        "{expected_tag_object_sha1}": (
            prepared.expected_tag_object_sha1
        ),
        "{verified_SystemRoot}": transport._environment["SystemRoot"],
        "{verified_WINDIR}": transport._environment["WINDIR"],
        "{verified_COMSPEC}": transport._environment["COMSPEC"],
        "{verified_transport_PATH}": transport._environment["PATH"],
        "{verified_PATHEXT}": transport._environment["PATHEXT"],
        "{verified_TEMP}": transport._environment["TEMP"],
        "{verified_TMP}": transport._environment["TMP"],
        "{verified_USERPROFILE}": transport._environment["USERPROFILE"],
        "{verified_LOCALAPPDATA}": transport._environment["LOCALAPPDATA"],
        "{verified_APPDATA}": transport._environment["APPDATA"],
        "{verified_HOME}": transport._environment["HOME"],
        "{verified_git_exec_path}": transport._environment["GIT_EXEC_PATH"],
    }

    def resolve(value: Any) -> Any:
        if type(value) is str:
            resolved = value
            for placeholder, replacement in replacements.items():
                resolved = resolved.replace(placeholder, replacement)
            return resolved
        if type(value) is list:
            return [resolve(item) for item in value]
        if type(value) is dict:
            return {key: resolve(item) for key, item in value.items()}
        return copy.deepcopy(value)

    return resolve(dict(template))


def _execution_from_executor(
    executor: Callable[..., PublicationProcessExecution],
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    profile_kind: str,
    transport_manifest_sha256: str,
    publication_intent_sha256: str,
    operation_kind: str,
    operation_sha256: str,
    worker_ownership_sha256: str,
    pre_push_authorization_sha256: str | None,
    expected_tag_object_sha1: str,
    tag_ref: str,
) -> PublicationProcessExecution:
    try:
        authorized = getattr(executor, "execute_authorized", None)
        if callable(authorized):
            request = _ContainedPublicationExecutionRequest(
                argv=argv,
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout_seconds,
                profile_kind=profile_kind,
                transport_manifest_sha256=transport_manifest_sha256,
                publication_intent_sha256=publication_intent_sha256,
                operation_kind=operation_kind,
                operation_sha256=operation_sha256,
                worker_ownership_sha256=worker_ownership_sha256,
                pre_push_authorization_sha256=(
                    pre_push_authorization_sha256
                ),
                expected_tag_object_sha1=expected_tag_object_sha1,
                tag_ref=tag_ref,
                _sentinel=_CONTAINED_EXECUTION_REQUEST_SENTINEL,
            )
            result = authorized(request)
        else:
            result = executor(
                argv=argv,
                cwd=cwd,
                env=dict(environment),
                timeout_seconds=timeout_seconds,
            )
    except subprocess.TimeoutExpired:
        return PublicationProcessExecution(
            process_exit_status="deadline",
            process_exit_code=None,
            stdout=b"",
            stderr=b"",
        )
    except OSError:
        return PublicationProcessExecution(
            process_exit_status="spawn_failed",
            process_exit_code=None,
            stdout=b"",
            stderr=b"",
        )
    if type(result) is not PublicationProcessExecution:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Contained process executor returned an unsupported result"
        )
    return result


def _timeout(value: Any) -> float:
    if type(value) not in {int, float}:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Publication timeout must be numeric"
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 300.0:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Publication timeout must be finite and at most 300 seconds"
        )
    return result


def _parse_remote_rows(
    stdout: bytes,
    *,
    tag_ref: str,
    expected_object: str,
    expected_peeled: str,
    expected_message_sha256: str,
) -> tuple[str, str, str, str, str]:
    matches = list(_REMOTE_ROW_RE.finditer(stdout))
    if (
        not stdout
        or b"\r" in stdout
        or b"\0" in stdout
        or b"".join(match.group(0) for match in matches) != stdout
        or len(matches) not in {1, 2}
    ):
        return (
            "protocol_error",
            "unknown",
            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
        )
    rows: dict[str, str] = {}
    requested = {tag_ref, f"{tag_ref}^{{}}"}
    for match in matches:
        ref_bytes = match.group("ref")
        try:
            ref = ref_bytes.decode("ascii", errors="strict")
            oid = match.group("oid").decode("ascii", errors="strict")
        except UnicodeDecodeError:
            return (
                "protocol_error",
                "unknown",
                PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
                PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
                PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
            )
        if ref not in requested or ref in rows:
            return (
                "protocol_error",
                "unknown",
                PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
                PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
                PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
            )
        rows[ref] = oid
    if tag_ref not in rows:
        return (
            "protocol_error",
            "unknown",
            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
            PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL,
        )
    object_value = rows[tag_ref]
    peeled_value = rows.get(
        f"{tag_ref}^{{}}",
        PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL,
    )
    message_value = (
        expected_message_sha256
        if object_value == expected_object
        else PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
    )
    return (
        "completed",
        "present",
        object_value,
        peeled_value,
        message_value,
    )


class PublicationRemoteReadbackResult:
    """Opaque exact command evidence and optional observation projection."""

    __slots__ = ("_evidence", "_observation", "_stdout", "_stderr", "_sentinel")

    def __init__(
        self,
        *,
        evidence: Mapping[str, Any],
        observation: Mapping[str, Any] | None,
        stdout: bytes,
        stderr: bytes,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _READBACK_RESULT_SENTINEL:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Readback results can only be issued by this module"
            )
        self._evidence = copy.deepcopy(dict(evidence))
        self._observation = (
            None
            if observation is None
            else copy.deepcopy(dict(observation))
        )
        self._stdout = bytes(stdout)
        self._stderr = bytes(stderr)
        self._sentinel = _sentinel

    @property
    def evidence(self) -> dict[str, Any]:
        return copy.deepcopy(self._evidence)

    @property
    def observation(self) -> dict[str, Any] | None:
        return (
            None
            if self._observation is None
            else copy.deepcopy(self._observation)
        )

    @property
    def stdout(self) -> bytes:
        return bytes(self._stdout)

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    @property
    def observation_permitted(self) -> bool:
        return self._observation is not None

    @property
    def observed_ref_state(self) -> str | None:
        if self._observation is None:
            return None
        return self._observation["observed_ref_state"]

    def __repr__(self) -> str:
        return "PublicationRemoteReadbackResult(<redacted>)"


class PublicationPushCommandResult:
    """Opaque evidence for the operation's sole authorized push command."""

    __slots__ = (
        "_command_sha256",
        "_execution",
        "_stdout_sha256",
        "_stderr_sha256",
        "_sentinel",
    )

    def __init__(
        self,
        *,
        command_sha256: str,
        execution: PublicationProcessExecution,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _PUSH_RESULT_SENTINEL:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Push results can only be issued by this module"
            )
        self._command_sha256 = command_sha256
        self._execution = execution
        self._stdout_sha256 = _bytes_sha256(execution.stdout)
        self._stderr_sha256 = _bytes_sha256(execution.stderr)
        self._sentinel = _sentinel

    @property
    def push_command_sha256(self) -> str:
        return self._command_sha256

    @property
    def process_exit_status(self) -> str:
        return self._execution.process_exit_status

    @property
    def process_exit_code(self) -> int | None:
        return self._execution.process_exit_code

    @property
    def stdout(self) -> bytes:
        return bytes(self._execution.stdout)

    @property
    def stderr(self) -> bytes:
        return bytes(self._execution.stderr)

    @property
    def push_command_count_upper_bound(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "PublicationPushCommandResult(<redacted>)"


def _validate_operation_context(
    *,
    transport: PreparedPublicationTransport,
    store_instance_id: str,
    store_session_nonce_sha256: str,
    publication_intent_sha256: str,
    operation_kind: str,
    operation_sha256: str,
    worker_ownership: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str]:
    manifest = _validate_transport_manifest(transport._manifest)
    store_id = _sha256(store_instance_id, "store instance ID")
    session = _sha256(store_session_nonce_sha256, "store session nonce")
    intent = _sha256(publication_intent_sha256, "publication intent")
    kind, operation = _validate_operation(
        operation_kind=operation_kind,
        operation_sha256=operation_sha256,
    )
    owner = _sha256(
        worker_ownership["worker_ownership_sha256"],
        "worker ownership",
    )
    if (
        manifest["store_instance_id"] != store_id
        or manifest["store_session_nonce_sha256"] != session
        or manifest["publication_intent_sha256"] != intent
        or manifest["operation_kind"] != kind
        or manifest["operation_sha256"] != operation
        or worker_ownership["contract_version"] != CONTRACT_VERSION
        or worker_ownership["contract_sha256"] != CONTRACT_SHA256
        or worker_ownership["implementation_manifest_sha256"]
        != manifest["implementation_manifest_sha256"]
        or worker_ownership["implementation_commit"]
        != manifest["implementation_commit"]
        or worker_ownership["store_instance_id"] != store_id
        or worker_ownership["store_session_nonce_sha256"] != session
        or worker_ownership["attempt_id"] != manifest["attempt_id"]
        or worker_ownership["publication_intent_sha256"] != intent
        or worker_ownership["operation_kind"] != kind
        or worker_ownership["operation_sha256"] != operation
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Publication operation crossed its isolated transport binding"
        )
    return store_id, session, intent, kind, operation, owner


def _validated_durable_transport_manifest(
    *,
    transport: PreparedPublicationTransport,
    authority: Any,
    validator: Callable[[Any], Mapping[str, Any]],
) -> dict[str, Any]:
    if not callable(validator):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Durable transport manifest validator must be callable"
        )
    observed = _validate_transport_manifest(validator(authority))
    if observed != transport.manifest:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Durable transport authority differs from the prepared manifest"
        )
    return observed


def _validated_worker_ownership(
    *,
    authority: Any,
    validator: Callable[[Any], Mapping[str, Any]],
) -> dict[str, Any]:
    if not callable(validator):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Worker ownership validator must be callable"
        )
    observed = _mapping(
        validator(authority),
        "validated worker ownership",
    )
    _expect_fields(
        observed,
        PUBLICATION_WORKER_OWNERSHIP_FIELDS,
        "worker ownership",
    )
    digest = _sha256(
        observed["worker_ownership_sha256"],
        "worker ownership self-hash",
    )
    if canonical_sha256(
        {
            key: value
            for key, value in observed.items()
            if key != "worker_ownership_sha256"
        }
    ) != digest:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Worker ownership self-hash is inconsistent"
        )
    if (
        observed["schema_version"]
        != (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-worker-ownership-v1"
        )
        or observed["owner_verifier_id"]
        != (
            "sec-gemma-online-risk-overlay-v2-2-"
            "publication-worker-owner-verifier-v1"
        )
        or observed["kill_on_parent_exit"] is not True
        or observed["child_assignment_before_resume_required"] is not True
        or observed["ownership_status"] != "claimed"
        or type(observed["owner_process_id"]) is not int
        or observed["owner_process_id"] <= 0
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Worker ownership fixed values changed"
        )
    return observed


class ExternalGitTagPublisher:
    """Execute exact readback and one-push profiles under runner containment."""

    __slots__ = (
        "_executor",
        "_consumed_authorizations",
        "_final_repo_root",
        "_final_implementation",
        "_final_clock",
        "_final_allow_url_rewrite",
        "_final_subprocess_runner",
        "_final_git_executable",
        "_final_git_sha256",
        "_final_git_root",
        "_final_dependency_closure_sha256",
        "_final_config_snapshot_sha256",
        "_final_test_url_rewrite_target",
        "_final_test_url_rewrite_target_identity_sha256",
        "_final_environment",
        "_final_no_hook_path",
    )

    def __init__(
        self,
        *,
        process_executor: (
            Callable[..., PublicationProcessExecution] | None
        ) = None,
        repo_root: Path | None = None,
        implementation_manifest: Mapping[str, Any] | None = None,
        clock: Any = time.monotonic,
        test_only_allow_url_rewrite: bool = False,
        test_only_url_rewrite_target: Path | None = None,
        final_registry_subprocess_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        if process_executor is not None and not callable(process_executor):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Publisher contained process executor must be callable"
            )
        self._executor = process_executor
        self._consumed_authorizations: set[str] = set()
        self._final_repo_root: Path | None = None
        self._final_implementation: dict[str, Any] | None = None
        self._final_clock = clock
        self._final_allow_url_rewrite = test_only_allow_url_rewrite
        self._final_subprocess_runner = final_registry_subprocess_runner
        self._final_git_executable: str | None = None
        self._final_git_sha256: str | None = None
        self._final_git_root: Path | None = None
        self._final_dependency_closure_sha256: str | None = None
        self._final_config_snapshot_sha256: str | None = None
        self._final_test_url_rewrite_target: Path | None = None
        self._final_test_url_rewrite_target_identity_sha256: str | None = None
        self._final_environment: dict[str, str] | None = None
        self._final_no_hook_path: Path | None = None
        final_mode_requested = (
            repo_root is not None or implementation_manifest is not None
        )
        if not final_mode_requested:
            if process_executor is None:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Publisher requires contained transport or final-registry mode"
                )
            return
        if (
            not isinstance(repo_root, Path)
            or not repo_root.is_absolute()
            or implementation_manifest is None
            or not callable(clock)
            or type(test_only_allow_url_rewrite) is not bool
            or (
                not test_only_allow_url_rewrite
                and test_only_url_rewrite_target is not None
            )
            or not callable(final_registry_subprocess_runner)
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry publisher configuration is incomplete"
            )
        try:
            resolved = repo_root.resolve(strict=True)
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry repository root is unavailable"
            ) from exc
        git_directory = resolved / ".git"
        if (
            not resolved.is_dir()
            or git_directory.is_symlink()
            or not git_directory.is_dir()
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry publication requires one non-worktree repository"
            )
        test_rewrite_target: Path | None = None
        test_rewrite_target_identity: str | None = None
        test_rewrite_uri: str | None = None
        if test_only_allow_url_rewrite:
            if (
                not isinstance(test_only_url_rewrite_target, Path)
                or not test_only_url_rewrite_target.is_absolute()
            ):
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Test-only final-registry rewrite requires one explicit target"
                )
            try:
                test_rewrite_target = (
                    test_only_url_rewrite_target.resolve(strict=True)
                )
            except OSError as exc:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Test-only final-registry rewrite target is unavailable"
                ) from exc
            if (
                test_rewrite_target != test_only_url_rewrite_target
                or test_rewrite_target.parent != resolved.parent
            ):
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Test-only final-registry rewrite target escaped its fixture"
                )
            test_rewrite_target_identity = (
                _test_only_bare_repo_identity_sha256(test_rewrite_target)
            )
            test_rewrite_uri = test_rewrite_target.as_uri()
        config_snapshot_sha256 = _final_registry_config_snapshot_sha256(
            git_directory,
            test_only_rewrite_uri=test_rewrite_uri,
        )
        try:
            credential_helper = Path(
                FROZEN_CREDENTIAL_HELPER_PATH
            ).resolve(strict=True)
            git_root = credential_helper.parents[2]
            git_path = (
                git_root / "mingw64" / "bin" / "git.exe"
            ).resolve(strict=True)
        except (IndexError, OSError) as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry real Git executable is unavailable"
            ) from exc
        no_hook_path = (
            git_directory
            / "sec-gemma-online-risk-overlay-v2-2-final-registry-disabled-hooks"
        )
        try:
            no_hook_path.mkdir(exist_ok=True)
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry no-hook directory could not be created"
            ) from exc
        selected_environment: dict[str, str] = {}
        for name in _HOST_ENVIRONMENT_KEYS:
            value = os.environ.get(name)
            if value:
                selected_environment[name] = value
        if "SystemRoot" not in selected_environment:
            selected_environment["SystemRoot"] = str(
                Path.home().anchor or "C:/Windows"
            )
        selected_environment.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_SYSTEM": "NUL",
                "GIT_CONFIG_GLOBAL": "NUL",
                "GIT_CONFIG_COUNT": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "GCM_INTERACTIVE": "Never",
                "GIT_ALLOW_PROTOCOL": (
                    "https:file"
                    if test_only_allow_url_rewrite
                    else "https"
                ),
                "GIT_PROTOCOL_FROM_USER": "0",
                "LANG": "C",
                "LC_ALL": "C",
            }
        )
        selected_environment["PATH"] = _portable_path(git_path.parent)
        self._final_repo_root = resolved
        self._final_implementation = validate_implementation_manifest(
            implementation_manifest
        )
        self._final_git_executable = _portable_path(git_path)
        self._final_git_sha256 = _file_sha256(git_path)
        self._final_git_root = git_root
        self._final_dependency_closure_sha256 = (
            git_runtime_dependency_closure_sha256(git_root)
        )
        self._final_config_snapshot_sha256 = config_snapshot_sha256
        self._final_test_url_rewrite_target = test_rewrite_target
        self._final_test_url_rewrite_target_identity_sha256 = (
            test_rewrite_target_identity
        )
        self._final_environment = selected_environment
        self._final_no_hook_path = no_hook_path

    def _final_remaining(self, deadline_monotonic: float) -> float:
        if (
            type(deadline_monotonic) not in {int, float}
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry deadline must be finite"
            )
        try:
            now = float(self._final_clock())
        except Exception as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry monotonic clock failed"
            ) from exc
        remaining = float(deadline_monotonic) - now
        if not math.isfinite(now) or now < 0.0 or remaining <= 0.0:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry deadline has expired"
            )
        return min(remaining, 300.0)

    def _verify_final_registry_mode(self) -> None:
        if (
            self._final_repo_root is None
            or self._final_implementation is None
            or self._final_git_executable is None
            or self._final_git_sha256 is None
            or self._final_git_root is None
            or self._final_dependency_closure_sha256 is None
            or self._final_config_snapshot_sha256 is None
            or self._final_environment is None
            or self._final_no_hook_path is None
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry compatibility mode is not configured"
            )
        if self._final_allow_url_rewrite:
            if (
                self._final_test_url_rewrite_target is None
                or self._final_test_url_rewrite_target_identity_sha256 is None
                or not hmac.compare_digest(
                    _test_only_bare_repo_identity_sha256(
                        self._final_test_url_rewrite_target
                    ),
                    self._final_test_url_rewrite_target_identity_sha256,
                )
            ):
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Test-only final-registry rewrite target identity changed"
                )
        elif (
            self._final_test_url_rewrite_target is not None
            or self._final_test_url_rewrite_target_identity_sha256 is not None
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Production final-registry publisher retained a test rewrite"
            )
        if not hmac.compare_digest(
            _file_sha256(Path(self._final_git_executable)),
            self._final_git_sha256,
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry Git executable changed"
            )
        if not hmac.compare_digest(
            git_runtime_dependency_closure_sha256(self._final_git_root),
            self._final_dependency_closure_sha256,
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry Git dependency closure changed"
            )
        if not hmac.compare_digest(
            _final_registry_config_snapshot_sha256(
                self._final_repo_root / ".git",
                test_only_rewrite_uri=(
                    self._final_test_url_rewrite_target.as_uri()
                    if self._final_test_url_rewrite_target is not None
                    else None
                ),
            ),
            self._final_config_snapshot_sha256,
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local/worktree Git configuration changed"
            )
        try:
            entries = list(self._final_no_hook_path.iterdir())
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry no-hook directory is unavailable"
            ) from exc
        if self._final_no_hook_path.is_symlink() or entries:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry no-hook directory is not exact and empty"
            )

    def _final_run(
        self,
        *args: str,
        deadline_monotonic: float,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> Any:
        self._verify_final_registry_mode()
        assert self._final_git_executable is not None
        assert self._final_repo_root is not None
        assert self._final_environment is not None
        assert self._final_no_hook_path is not None
        command = [
            self._final_git_executable,
            "-c",
            f"core.hooksPath={_portable_path(self._final_no_hook_path)}",
            "-c",
            "tag.gpgSign=false",
            "-c",
            "credential.interactive=never",
            "-c",
            "credential.helper=",
            "-c",
            f"credential.helper={FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE}",
            "-c",
            "credential.useHttpPath=true",
            "-c",
            "http.followRedirects=false",
            "-c",
            "http.sslVerify=true",
            *args,
        ]
        try:
            completed = self._final_subprocess_runner(
                command,
                cwd=self._final_repo_root,
                env=dict(self._final_environment),
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self._final_remaining(deadline_monotonic),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry Git command failed or exceeded its deadline"
            ) from exc
        if (
            not hasattr(completed, "returncode")
            or not hasattr(completed, "stdout")
            or not hasattr(completed, "stderr")
            or type(completed.returncode) is not int
            or type(completed.stdout) is not bytes
            or type(completed.stderr) is not bytes
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry Git runner returned malformed evidence"
            )
        if check and completed.returncode != 0:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry Git command failed closed"
            )
        return completed

    def _final_text(
        self,
        *args: str,
        deadline_monotonic: float,
        input_bytes: bytes | None = None,
    ) -> str:
        completed = self._final_run(
            *args,
            deadline_monotonic=deadline_monotonic,
            input_bytes=input_bytes,
        )
        try:
            return completed.stdout.decode(
                "utf-8",
                errors="strict",
            ).strip()
        except UnicodeDecodeError as exc:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry Git output is not strict UTF-8"
            ) from exc

    def _verify_final_registry_repository(
        self,
        *,
        deadline_monotonic: float,
    ) -> None:
        self._verify_final_registry_mode()
        assert self._final_implementation is not None
        implementation_commit = self._final_implementation[
            "implementation_commit"
        ]
        if self._final_text(
            "rev-parse",
            "--verify",
            "HEAD",
            deadline_monotonic=deadline_monotonic,
        ) != implementation_commit:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry HEAD differs from the implementation"
            )
        if self._final_text(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            deadline_monotonic=deadline_monotonic,
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry repository is not clean"
            )
        origin = self._final_text(
            "config",
            "--get",
            "remote.origin.url",
            deadline_monotonic=deadline_monotonic,
        )
        if origin != ALLOWED_REMOTE_URL:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry origin URL changed"
            )
        push_urls = self._final_run(
            "config",
            "--get-all",
            "remote.origin.pushurl",
            deadline_monotonic=deadline_monotonic,
            check=False,
        )
        rewrites = self._final_run(
            "config",
            "--get-regexp",
            r"^url\..*\.(insteadof|pushinsteadof)$",
            deadline_monotonic=deadline_monotonic,
            check=False,
        )
        if (
            push_urls.returncode not in {0, 1}
            or rewrites.returncode not in {0, 1}
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry endpoint configuration check failed"
            )
        if (
            push_urls.stdout.strip() or rewrites.stdout.strip()
        ) and not self._final_allow_url_rewrite:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry endpoint is redirected or rewritten"
            )
        branch_ref = (
            f"refs/heads/{self._final_implementation['branch']}"
        )
        remote_branch = self._final_text(
            "ls-remote",
            ALLOWED_REMOTE_URL,
            branch_ref,
            deadline_monotonic=deadline_monotonic,
        )
        if remote_branch != f"{implementation_commit}\t{branch_ref}":
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry remote branch differs from the implementation"
            )

    def _final_materialize_tag(
        self,
        *,
        prepared: PreparedExternalPublication,
        deadline_monotonic: float,
    ) -> str:
        observed = self._final_text(
            "mktag",
            deadline_monotonic=deadline_monotonic,
            input_bytes=prepared.tag_object_bytes,
        )
        if observed != prepared.expected_tag_object_sha1:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry local tag object differs from its precomputation"
            )
        return observed

    def _final_read_remote_tag(
        self,
        *,
        prepared: PreparedExternalPublication,
        deadline_monotonic: float,
    ) -> tuple[str, str] | None:
        output = self._final_run(
            "ls-remote",
            "--tags",
            ALLOWED_REMOTE_URL,
            prepared.tag_ref,
            f"{prepared.tag_ref}^{{}}",
            deadline_monotonic=deadline_monotonic,
        )
        if output.stderr:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry readback produced unexpected stderr"
            )
        if output.stdout == b"":
            return None
        (
            transport_status,
            lookup_status,
            object_value,
            peeled_value,
            _,
        ) = _parse_remote_rows(
            output.stdout,
            tag_ref=prepared.tag_ref,
            expected_object=prepared.expected_tag_object_sha1,
            expected_peeled=prepared.expected_peeled_commit,
            expected_message_sha256=canonical_sha256(
                prepared.tag_message
            ),
        )
        if transport_status != "completed" or lookup_status != "present":
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry remote tag readback is malformed"
            )
        return object_value, peeled_value

    def _final_registry_prepared(
        self,
        *,
        attempt_id: str,
        terminal_status: str,
        report_kind: str,
        artifact_sha256: str,
        predecessor_publication_sha256: str,
    ) -> PreparedExternalPublication:
        self._verify_final_registry_mode()
        assert self._final_implementation is not None
        if (
            attempt_id != FINAL_ATTEMPT_ID
            or terminal_status != REGISTERED_UNRUN
            or report_kind != FINAL_REGISTRY_SUCCESSOR
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Compatibility publication is restricted to the exact final registry"
            )
        return prepare_external_publication(
            implementation_manifest=self._final_implementation,
            attempt_id=attempt_id,
            terminal_status=terminal_status,
            report_kind=report_kind,
            artifact_sha256=artifact_sha256,
            predecessor_publication_sha256=(
                predecessor_publication_sha256
            ),
        )

    def _publish_final_registry_once(
        self,
        *,
        prepared: PreparedExternalPublication,
        deadline_monotonic: float,
    ) -> VerifiedExternalPublication:
        self._verify_final_registry_repository(
            deadline_monotonic=deadline_monotonic
        )
        prior = self._final_read_remote_tag(
            prepared=prepared,
            deadline_monotonic=deadline_monotonic,
        )
        if prior is not None:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry stable ref already exists"
            )
        local_object = self._final_materialize_tag(
            prepared=prepared,
            deadline_monotonic=deadline_monotonic,
        )
        self._final_run(
            "push",
            "--porcelain",
            "--no-verify",
            ALLOWED_REMOTE_URL,
            f"{local_object}:{prepared.tag_ref}",
            deadline_monotonic=deadline_monotonic,
        )
        observed = self._final_read_remote_tag(
            prepared=prepared,
            deadline_monotonic=deadline_monotonic,
        )
        if observed != (
            prepared.expected_tag_object_sha1,
            prepared.expected_peeled_commit,
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Final-registry post-push readback differs from the expected tag"
            )
        return _issue_verified_external_publication(
            prepared.expected_publication,
            implementation_manifest=prepared.implementation_manifest,
        )

    def _recover_final_registry(
        self,
        *,
        prepared: PreparedExternalPublication,
        deadline_monotonic: float,
    ) -> VerifiedExternalPublication:
        self._verify_final_registry_repository(
            deadline_monotonic=deadline_monotonic
        )
        self._final_materialize_tag(
            prepared=prepared,
            deadline_monotonic=deadline_monotonic,
        )
        observed = self._final_read_remote_tag(
            prepared=prepared,
            deadline_monotonic=deadline_monotonic,
        )
        if observed != (
            prepared.expected_tag_object_sha1,
            prepared.expected_peeled_commit,
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Existing final-registry tag is not exactly recoverable"
            )
        return _issue_verified_external_publication(
            prepared.expected_publication,
            implementation_manifest=prepared.implementation_manifest,
        )

    def publish_or_recover_final_registry(
        self,
        *,
        attempt_id: str,
        terminal_status: str,
        report_kind: str,
        artifact_sha256: str,
        predecessor_publication_sha256: str,
        deadline_monotonic: float,
    ) -> VerifiedExternalPublication:
        """Publish or read-only recover only the exact final-registry tag."""

        prepared = self._final_registry_prepared(
            attempt_id=attempt_id,
            terminal_status=terminal_status,
            report_kind=report_kind,
            artifact_sha256=artifact_sha256,
            predecessor_publication_sha256=(
                predecessor_publication_sha256
            ),
        )
        try:
            return self._publish_final_registry_once(
                prepared=prepared,
                deadline_monotonic=deadline_monotonic,
            )
        except SecGemmaOnlineRiskOverlayPublisherError:
            try:
                return self._recover_final_registry(
                    prepared=prepared,
                    deadline_monotonic=deadline_monotonic,
                )
            except SecGemmaOnlineRiskOverlayPublisherError as exc:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Final-registry tag is neither new nor exactly recoverable"
                ) from exc

    def read_remote(
        self,
        *,
        prepared_publication: PreparedExternalPublication,
        transport: PreparedPublicationTransport,
        durable_transport_manifest: Any,
        transport_manifest_validator: Callable[
            [Any], Mapping[str, Any]
        ],
        store_instance_id: str,
        store_session_nonce_sha256: str,
        publication_intent_sha256: str,
        operation_kind: str,
        operation_sha256: str,
        worker_ownership: Any,
        worker_ownership_validator: Callable[
            [Any], Mapping[str, Any]
        ],
        observation_ordinal: int,
        observation_phase: str,
        prior_push_command_sha256: str,
        pre_observation_store_journal_sequence: int,
        pre_observation_store_journal_tip_sha256: str,
        timeout_seconds: float,
    ) -> PublicationRemoteReadbackResult:
        """Run exactly one raw-byte ls-remote and classify it fail-closed."""

        if self._executor is None:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Ordinary readback requires the runner's contained executor"
            )
        _verify_transport(transport, prepared_publication)
        durable_manifest = _validated_durable_transport_manifest(
            transport=transport,
            authority=durable_transport_manifest,
            validator=transport_manifest_validator,
        )
        owner_material = _validated_worker_ownership(
            authority=worker_ownership,
            validator=worker_ownership_validator,
        )
        (
            store_id,
            session,
            intent,
            fixed_kind,
            fixed_operation,
            owner,
        ) = _validate_operation_context(
            transport=transport,
            store_instance_id=store_instance_id,
            store_session_nonce_sha256=store_session_nonce_sha256,
            publication_intent_sha256=publication_intent_sha256,
            operation_kind=operation_kind,
            operation_sha256=operation_sha256,
            worker_ownership=owner_material,
        )
        ordinal = _integer(
            observation_ordinal,
            "observation ordinal",
            minimum=1,
        )
        if observation_phase == PRE_PUSH:
            if (
                ordinal != 1
                or prior_push_command_sha256
                != PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
            ):
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Pre-push observation phase or prior command changed"
                )
        elif observation_phase == POST_PUSH:
            if ordinal != 2:
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Post-push observation must have ordinal two"
                )
            _sha256(prior_push_command_sha256, "prior push command")
            if (
                prior_push_command_sha256
                == PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
            ):
                raise SecGemmaOnlineRiskOverlayPublisherError(
                    "Post-push observation lacks a push command"
                )
        else:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Observation phase is not frozen"
            )
        pre_sequence = _integer(
            pre_observation_store_journal_sequence,
            "pre-observation journal sequence",
        )
        pre_tip = _sha256(
            pre_observation_store_journal_tip_sha256,
            "pre-observation journal tip",
        )
        profile = _resolved_profile(
            template=_readback_contract_profile(),
            transport=transport,
            prepared=prepared_publication,
        )
        command_sequence_sha256 = canonical_sha256(profile)
        argv = (
            transport._pins["git_executable_path"],
            *profile["argv_template"],
        )
        execution = _execution_from_executor(
            self._executor,
            argv=argv,
            cwd=transport.directory,
            environment=transport._environment,
            timeout_seconds=_timeout(timeout_seconds),
            profile_kind="readback",
            transport_manifest_sha256=durable_manifest[
                "isolated_transport_git_directory_manifest_sha256"
            ],
            publication_intent_sha256=intent,
            operation_kind=fixed_kind,
            operation_sha256=fixed_operation,
            worker_ownership_sha256=owner,
            pre_push_authorization_sha256=None,
            expected_tag_object_sha1=(
                prepared_publication.expected_tag_object_sha1
            ),
            tag_ref=prepared_publication.tag_ref,
        )
        stdout = execution.stdout
        stderr = execution.stderr
        raw_object = PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
        raw_peeled = PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
        raw_message = PUBLICATION_REMOTE_VALUE_MISSING_SENTINEL
        if execution.process_exit_status != "exited":
            transport_status = "unavailable"
            lookup_status = "unknown"
        elif execution.process_exit_code != 0:
            transport_status = "unavailable"
            lookup_status = "unknown"
        elif stderr:
            transport_status = "protocol_error"
            lookup_status = "unknown"
            raw_object = PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL
            raw_peeled = PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL
            raw_message = PUBLICATION_REMOTE_VALUE_MALFORMED_SENTINEL
        elif stdout == b"":
            transport_status = "completed"
            lookup_status = "absent"
            raw_object = PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
            raw_peeled = PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
            raw_message = PUBLICATION_REMOTE_REF_ABSENT_SENTINEL
        else:
            (
                transport_status,
                lookup_status,
                raw_object,
                raw_peeled,
                raw_message,
            ) = _parse_remote_rows(
                stdout,
                tag_ref=prepared_publication.tag_ref,
                expected_object=(
                    prepared_publication.expected_tag_object_sha1
                ),
                expected_peeled=(
                    prepared_publication.expected_peeled_commit
                ),
                expected_message_sha256=canonical_sha256(
                    prepared_publication.tag_message
                ),
            )
        evidence_body = {
            "schema_version": READBACK_EVIDENCE_SCHEMA_VERSION,
            "evidence_verifier_id": READBACK_EVIDENCE_VERIFIER_ID,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "implementation_manifest_sha256": (
                prepared_publication.implementation_manifest[
                    "implementation_manifest_sha256"
                ]
            ),
            "implementation_commit": (
                prepared_publication.expected_peeled_commit
            ),
            "store_instance_id": store_id,
            "store_session_nonce_sha256": session,
            "attempt_id": prepared_publication.tag_message["attempt_id"],
            "publication_intent_sha256": intent,
            "observation_operation_kind": fixed_kind,
            "observation_operation_sha256": fixed_operation,
            "worker_ownership_sha256": owner,
            "observation_ordinal": ordinal,
            "observation_phase": observation_phase,
            "prior_push_command_sha256": prior_push_command_sha256,
            "tag_ref": prepared_publication.tag_ref,
            "remote_name": REMOTE_NAME,
            "remote_url": ALLOWED_REMOTE_URL,
            "transport_isolation_profile_sha256": (
                TRANSPORT_ISOLATION_PROFILE_SHA256
            ),
            "isolated_transport_git_directory_manifest_sha256": (
                transport.manifest_sha256
            ),
            "command_profile_id": READBACK_COMMAND_PROFILE_ID,
            "command_sequence_sha256": command_sequence_sha256,
            "process_exit_status": execution.process_exit_status,
            "process_exit_code": execution.process_exit_code,
            "stdout_byte_count": len(stdout),
            "stdout_sha256": _bytes_sha256(stdout),
            "stderr_byte_count": len(stderr),
            "stderr_sha256": _bytes_sha256(stderr),
            "transport_status": transport_status,
            "ref_lookup_status": lookup_status,
            "raw_ref_object_value": raw_object,
            "raw_peeled_value": raw_peeled,
            "raw_tag_message_sha256": raw_message,
        }
        evidence = {
            **evidence_body,
            "remote_readback_evidence_sha256": canonical_sha256(
                evidence_body
            ),
        }
        _expect_fields(
            evidence,
            PUBLICATION_REMOTE_READBACK_EVIDENCE_FIELDS,
            "remote readback evidence",
        )
        observation: dict[str, Any] | None = None
        if transport_status == "completed" and lookup_status in {
            "absent",
            "present",
        }:
            if lookup_status == "absent":
                observed_state = "absent"
            elif (
                raw_object
                == prepared_publication.expected_tag_object_sha1
                and raw_peeled
                == prepared_publication.expected_peeled_commit
                and raw_message
                == canonical_sha256(prepared_publication.tag_message)
            ):
                observed_state = "exact_expected"
            else:
                observed_state = "conflicting"
            observation_body = {
                "schema_version": REMOTE_OBSERVATION_SCHEMA_VERSION,
                "observation_verifier_id": (
                    REMOTE_OBSERVATION_VERIFIER_ID
                ),
                "contract_version": CONTRACT_VERSION,
                "contract_sha256": CONTRACT_SHA256,
                "implementation_manifest_sha256": (
                    prepared_publication.implementation_manifest[
                        "implementation_manifest_sha256"
                    ]
                ),
                "implementation_commit": (
                    prepared_publication.expected_peeled_commit
                ),
                "store_instance_id": store_id,
                "store_session_nonce_sha256": session,
                "attempt_id": prepared_publication.tag_message["attempt_id"],
                "publication_intent_sha256": intent,
                "observation_operation_kind": fixed_kind,
                "observation_operation_sha256": fixed_operation,
                "worker_ownership_sha256": owner,
                "observation_ordinal": ordinal,
                "observation_phase": observation_phase,
                "prior_push_command_sha256": prior_push_command_sha256,
                "tag_ref": prepared_publication.tag_ref,
                "remote_name": REMOTE_NAME,
                "remote_url": ALLOWED_REMOTE_URL,
                "expected_tag_object_sha1": (
                    prepared_publication.expected_tag_object_sha1
                ),
                "expected_peeled_commit": (
                    prepared_publication.expected_peeled_commit
                ),
                "observed_ref_state": observed_state,
                "observed_tag_object_sha1": raw_object,
                "observed_peeled_commit": raw_peeled,
                "observed_tag_message_sha256": raw_message,
                "remote_readback_evidence_sha256": evidence[
                    "remote_readback_evidence_sha256"
                ],
                "pre_observation_store_journal_sequence": pre_sequence,
                "pre_observation_store_journal_tip_sha256": pre_tip,
            }
            observation = {
                **observation_body,
                "publication_remote_observation_sha256": canonical_sha256(
                    observation_body
                ),
            }
            _expect_fields(
                observation,
                PUBLICATION_REMOTE_OBSERVATION_FIELDS,
                "remote observation",
            )
        return PublicationRemoteReadbackResult(
            evidence=evidence,
            observation=observation,
            stdout=stdout,
            stderr=stderr,
            _sentinel=_READBACK_RESULT_SENTINEL,
        )

    def push_once(
        self,
        *,
        prepared_publication: PreparedExternalPublication,
        transport: PreparedPublicationTransport,
        pre_push_authorization: Any,
        authorization_validator: Callable[[Any], Mapping[str, Any]],
        absent_remote_observation: Any,
        observation_validator: Callable[[Any], Mapping[str, Any]],
        timeout_seconds: float,
    ) -> PublicationPushCommandResult:
        """Consume one durable absent-ref authorization and issue one push."""

        if self._executor is None:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Ordinary push requires the runner's contained executor"
            )
        _verify_transport(transport, prepared_publication)
        if not callable(authorization_validator) or not callable(
            observation_validator
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Push requires opaque store authority validators"
            )
        authorization = _mapping(
            authorization_validator(pre_push_authorization),
            "validated pre-push authorization",
        )
        observation = _mapping(
            observation_validator(absent_remote_observation),
            "validated absent remote observation",
        )
        _expect_fields(
            authorization,
            PUBLICATION_PRE_PUSH_AUTHORIZATION_FIELDS,
            "pre-push authorization",
        )
        _expect_fields(
            observation,
            PUBLICATION_REMOTE_OBSERVATION_FIELDS,
            "absent remote observation",
        )
        authorization_hash = _sha256(
            authorization["pre_push_authorization_sha256"],
            "pre-push authorization self-hash",
        )
        if canonical_sha256(
            {
                key: value
                for key, value in authorization.items()
                if key != "pre_push_authorization_sha256"
            }
        ) != authorization_hash:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Pre-push authorization self-hash is inconsistent"
            )
        observation_hash = _sha256(
            observation["publication_remote_observation_sha256"],
            "absent remote observation self-hash",
        )
        if canonical_sha256(
            {
                key: value
                for key, value in observation.items()
                if key != "publication_remote_observation_sha256"
            }
        ) != observation_hash:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Absent remote observation self-hash is inconsistent"
            )
        manifest = transport.manifest
        authorization_ordinal = authorization[
            "authorization_operation_ordinal"
        ]
        if (
            type(authorization_ordinal) is not int
            or (
                manifest["operation_kind"] == NORMAL_PUBLICATION
                and authorization_ordinal != 0
            )
            or (
                manifest["operation_kind"] == PUBLICATION_RECOVERY
                and authorization_ordinal < 1
            )
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Push authorization operation ordinal changed"
            )
        if (
            authorization["schema_version"]
            != (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-pre-push-authorization-v1"
            )
            or authorization["authorization_verifier_id"]
            != (
                "sec-gemma-online-risk-overlay-v2-2-"
                "publication-pre-push-authorization-verifier-v1"
            )
            or authorization["contract_version"] != CONTRACT_VERSION
            or authorization["contract_sha256"] != CONTRACT_SHA256
            or authorization["implementation_manifest_sha256"]
            != manifest["implementation_manifest_sha256"]
            or authorization["implementation_commit"]
            != manifest["implementation_commit"]
            or authorization["store_instance_id"]
            != manifest["store_instance_id"]
            or authorization["store_session_nonce_sha256"]
            != manifest["store_session_nonce_sha256"]
            or authorization["attempt_id"] != manifest["attempt_id"]
            or authorization["publication_intent_sha256"]
            != manifest["publication_intent_sha256"]
            or authorization["authorization_operation_kind"]
            != manifest["operation_kind"]
            or authorization["authorization_operation_sha256"]
            != manifest["operation_sha256"]
            or authorization["worker_ownership_sha256"]
            != observation["worker_ownership_sha256"]
            or authorization["remote_observation_sha256"]
            != observation_hash
            or authorization["tag_ref"] != prepared_publication.tag_ref
            or authorization["expected_tag_object_sha1"]
            != prepared_publication.expected_tag_object_sha1
            or authorization["authorization_status"]
            != "push_authorized_once"
            or authorization["push_command_limit"] != 1
            or observation["schema_version"]
            != REMOTE_OBSERVATION_SCHEMA_VERSION
            or observation["observation_verifier_id"]
            != REMOTE_OBSERVATION_VERIFIER_ID
            or observation["observed_ref_state"] != "absent"
            or observation["observation_phase"] != PRE_PUSH
            or observation["observation_ordinal"] != 1
            or observation["prior_push_command_sha256"]
            != PUBLICATION_NO_PRIOR_PUSH_COMMAND_SHA256
            or observation["publication_intent_sha256"]
            != manifest["publication_intent_sha256"]
            or observation["observation_operation_sha256"]
            != manifest["operation_sha256"]
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Push authorization crossed its durable absent observation"
            )
        if authorization_hash in self._consumed_authorizations:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Pre-push authorization was already consumed in this process"
            )
        self._consumed_authorizations.add(authorization_hash)
        profile = _resolved_profile(
            template=_push_contract_profile(),
            transport=transport,
            prepared=prepared_publication,
        )
        argv = (
            transport._pins["git_executable_path"],
            *profile["argv_template"],
        )
        if (
            "--force" in argv
            or "-f" in argv
            or "--no-verify" not in argv
            or ALLOWED_REMOTE_URL not in argv
            or REMOTE_NAME in argv
        ):
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Resolved push command is not the exact no-force literal URL command"
            )
        execution = _execution_from_executor(
            self._executor,
            argv=argv,
            cwd=transport.directory,
            environment=transport._environment,
            timeout_seconds=_timeout(timeout_seconds),
            profile_kind="push",
            transport_manifest_sha256=transport.manifest_sha256,
            publication_intent_sha256=authorization[
                "publication_intent_sha256"
            ],
            operation_kind=authorization[
                "authorization_operation_kind"
            ],
            operation_sha256=authorization[
                "authorization_operation_sha256"
            ],
            worker_ownership_sha256=authorization[
                "worker_ownership_sha256"
            ],
            pre_push_authorization_sha256=authorization_hash,
            expected_tag_object_sha1=(
                prepared_publication.expected_tag_object_sha1
            ),
            tag_ref=prepared_publication.tag_ref,
        )
        command_material = {
            "resolved_profile": profile,
            "process_exit_status": execution.process_exit_status,
            "process_exit_code": execution.process_exit_code,
            "stdout_byte_count": len(execution.stdout),
            "stdout_sha256": _bytes_sha256(execution.stdout),
            "stderr_byte_count": len(execution.stderr),
            "stderr_sha256": _bytes_sha256(execution.stderr),
        }
        return PublicationPushCommandResult(
            command_sha256=canonical_sha256(command_material),
            execution=execution,
            _sentinel=_PUSH_RESULT_SENTINEL,
        )


class VerifiedExternalPublication:
    """Opaque proof issued only from an exact durable remote observation."""

    __slots__ = ("_publication", "_sentinel")

    def __init__(
        self,
        *,
        publication: Mapping[str, Any],
        _sentinel: object,
    ) -> None:
        if _sentinel is not _VERIFIED_EXTERNAL_PUBLICATION_SENTINEL:
            raise SecGemmaOnlineRiskOverlayPublisherError(
                "Verified publications can only be issued by this module"
            )
        self._publication = copy.deepcopy(dict(publication))
        self._sentinel = _sentinel

    @property
    def publication(self) -> dict[str, Any]:
        return copy.deepcopy(self._publication)

    @property
    def publication_sha256(self) -> str:
        return self._publication["publication_sha256"]

    @property
    def artifact_sha256(self) -> str:
        return self._publication["artifact_sha256"]

    def __repr__(self) -> str:
        return "VerifiedExternalPublication(<redacted>)"


def is_verified_external_publication(value: Any) -> bool:
    return (
        type(value) is VerifiedExternalPublication
        and getattr(value, "_sentinel", None)
        is _VERIFIED_EXTERNAL_PUBLICATION_SENTINEL
    )


def _issue_verified_external_publication(
    publication: Mapping[str, Any],
    *,
    implementation_manifest: Mapping[str, Any],
) -> VerifiedExternalPublication:
    validated = validate_external_publication(
        publication,
        implementation_manifest=implementation_manifest,
    )
    return VerifiedExternalPublication(
        publication=validated,
        _sentinel=_VERIFIED_EXTERNAL_PUBLICATION_SENTINEL,
    )


def issue_verified_external_publication_from_observation(
    *,
    prepared_publication: PreparedExternalPublication,
    durable_remote_observation: Any,
    observation_validator: Callable[[Any], Mapping[str, Any]],
) -> VerifiedExternalPublication:
    """Issue or recover the same receipt from one exact durable observation."""

    if not is_prepared_external_publication(prepared_publication):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Publication receipt requires prepared deterministic material"
        )
    if not callable(observation_validator):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Publication observation validator must be callable"
        )
    observation = _mapping(
        observation_validator(durable_remote_observation),
        "validated durable remote observation",
    )
    _expect_fields(
        observation,
        PUBLICATION_REMOTE_OBSERVATION_FIELDS,
        "durable remote observation",
    )
    observation_hash = _sha256(
        observation["publication_remote_observation_sha256"],
        "durable remote observation self-hash",
    )
    if canonical_sha256(
        {
            key: value
            for key, value in observation.items()
            if key != "publication_remote_observation_sha256"
        }
    ) != observation_hash:
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Durable remote observation self-hash is inconsistent"
        )
    expected_message_hash = canonical_sha256(
        prepared_publication.tag_message
    )
    if (
        observation["schema_version"] != REMOTE_OBSERVATION_SCHEMA_VERSION
        or observation["observation_verifier_id"]
        != REMOTE_OBSERVATION_VERIFIER_ID
        or observation["contract_version"] != CONTRACT_VERSION
        or observation["contract_sha256"] != CONTRACT_SHA256
        or observation["implementation_manifest_sha256"]
        != prepared_publication.implementation_manifest[
            "implementation_manifest_sha256"
        ]
        or observation["implementation_commit"]
        != prepared_publication.expected_peeled_commit
        or observation["attempt_id"]
        != prepared_publication.tag_message["attempt_id"]
        or observation["tag_ref"] != prepared_publication.tag_ref
        or observation["remote_name"] != REMOTE_NAME
        or observation["remote_url"] != ALLOWED_REMOTE_URL
        or observation["expected_tag_object_sha1"]
        != prepared_publication.expected_tag_object_sha1
        or observation["expected_peeled_commit"]
        != prepared_publication.expected_peeled_commit
        or observation["observed_ref_state"] != "exact_expected"
        or observation["observed_tag_object_sha1"]
        != prepared_publication.expected_tag_object_sha1
        or observation["observed_peeled_commit"]
        != prepared_publication.expected_peeled_commit
        or observation["observed_tag_message_sha256"]
        != expected_message_hash
    ):
        raise SecGemmaOnlineRiskOverlayPublisherError(
            "Durable observation does not prove the exact expected tag"
        )
    return _issue_verified_external_publication(
        prepared_publication.expected_publication,
        implementation_manifest=(
            prepared_publication.implementation_manifest
        ),
    )


__all__ = [
    "ACQUISITION_PASS",
    "ALLOWED_REMOTE_URL",
    "EXTERNAL_PUBLICATION_SCHEMA_VERSION",
    "EXTERNAL_PUBLISHER_ID",
    "EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION",
    "ExternalGitTagPublisher",
    "FINAL_REGISTRY_SUCCESSOR",
    "FROZEN_CREDENTIAL_HELPER_CONFIG_VALUE",
    "NORMAL_PUBLICATION",
    "POST_PUSH",
    "PRE_PUSH",
    "PUBLICATION_GENESIS_SHA256",
    "PUBLICATION_RECOVERY",
    "PreparedExternalPublication",
    "PreparedPublicationTransport",
    "PublicationProcessExecution",
    "PublicationPushCommandResult",
    "PublicationRemoteReadbackResult",
    "REGISTERED_UNRUN",
    "REMOTE_NAME",
    "REPORT_KINDS",
    "SCORED_FAILED_GATE",
    "SCORED_PASS",
    "SecGemmaOnlineRiskOverlayPublisherError",
    "TERMINAL_FAIL",
    "TERMINAL_PASS",
    "TRANSPORT_ISOLATION_PROFILE_SHA256",
    "VerifiedExternalPublication",
    "build_exact_child_environment",
    "build_external_publication",
    "build_external_tag_message",
    "build_transport_executable_identity_pins",
    "git_runtime_dependency_closure_sha256",
    "is_prepared_external_publication",
    "is_prepared_publication_transport",
    "is_verified_external_publication",
    "issue_verified_external_publication_from_observation",
    "prepare_external_publication",
    "prepare_isolated_publication_transport",
    "reconstruct_prepared_external_publication",
    "validate_external_publication",
    "validate_external_tag_message",
    "validate_transport_executable_identity_pins",
    "verify_pinned_transport_executables",
]
