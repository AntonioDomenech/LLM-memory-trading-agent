"""Lean, resumable SEC/Gemma evidence runner.

The first supported command is a zero-effect preflight.  It verifies the
preregistered branch, the unchanged v2.2 scientific parent, the private SEC
contact fingerprint, the exact installed local model, and the ignored local
checkpoint area.  It never downloads SEC or market data and never calls model
generation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time
import uuid
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    MODEL_NAME,
    NEW_SOURCE_FILES,
    RUNTIME_FINGERPRINT_SHA256,
    SOURCE_PIN_FILES,
    canonical_sha256,
)


SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-lean-evidence-v3-preflight-v1"
BRANCH_NAME: Final[str] = "codex/aapl-sec-gemma-lean-evidence-v3"
PREREGISTRATION_COMMIT: Final[str] = (
    "f64e90e99dfb99f5b17609513c7d931dc939811f"
)
SCIENTIFIC_PARENT_COMMIT: Final[str] = (
    "efbc481c57e480d48303763163676e64e87df49d"
)
SCIENTIFIC_PARENT_TREE: Final[str] = (
    "bc91d619a4d04680d9600b1acd74f0a29662d0f9"
)
V22_PREREGISTRATION_COMMIT: Final[str] = (
    "a849b9d704ffd98547e570735a221b2b75f7db86"
)
V22_CONTRACT_MANIFEST_SHA256: Final[str] = (
    "64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d"
)
V22_IMPLEMENTATION_MANIFEST_SHA256: Final[str] = (
    "a779c0b669ff7fa323a81fd6a6ef7467ad3333caf35ef452d9fde2195d4d3804"
)
V22_DEPENDENCY_CLOSURE_SHA256: Final[str] = (
    "d37bb45d1d792efa88cb834e1b6a0cabde5ea3e435466a00905002b29152e1bd"
)
EXPECTED_ORIGIN_URL: Final[str] = (
    "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git"
)
PREREGISTRATION_PATH: Final[str] = "docs/aapl_sec_gemma_lean_evidence_v3.md"
PRIVATE_CONFIG_PATH: Final[str] = "data/local_config.json"
CHECKPOINT_ROOT: Final[str] = "data/aapl_sec_gemma_lean_evidence_v3"
PREFLIGHT_ARTIFACT_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_evidence_v3/PREFLIGHT.json"
)
ACTIVE_LOCK_NAME: Final[str] = "ACTIVE.lock"
RUNNER_SOURCE_PATH: Final[str] = "agent_benchmark/sec_gemma_lean_runner.py"
PACKAGE_INIT_SOURCE_PATH: Final[str] = "agent_benchmark/__init__.py"
MARKET_EVIDENCE_SOURCE_PATH: Final[str] = (
    "agent_benchmark/sec_filing_gemma_market_evidence.py"
)
MAX_PRIVATE_CONFIG_BYTES: Final[int] = 1024 * 1024

FROZEN_SCIENTIFIC_PATHS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            PACKAGE_INIT_SOURCE_PATH,
            "agent_benchmark/sec_gemma_online_risk_overlay_contract.py",
            MARKET_EVIDENCE_SOURCE_PATH,
            *(SOURCE_PIN_FILES.values()),
            *(NEW_SOURCE_FILES.values()),
        }
    )
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA256_TAGGED_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z\Z"
)
_REPARSE_FLAG: Final[int] = getattr(
    stat,
    "FILE_ATTRIBUTE_REPARSE_POINT",
    0x400,
)


class SecGemmaLeanRunnerError(RuntimeError):
    """A fixed, public-safe lean-runner check failed."""

    def __init__(self, code: str):
        if (
            type(code) is not str
            or not code
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code)
        ):
            code = "invalid_error_code"
        self.code = code
        super().__init__(code)

    def __repr__(self) -> str:
        return f"SecGemmaLeanRunnerError(code={self.code!r})"


@dataclass(frozen=True)
class _GitResult:
    returncode: int
    stdout: bytes


GitRunner = Callable[[Path, tuple[str, ...]], _GitResult]


def _default_git(repo_root: Path, arguments: tuple[str, ...]) -> _GitResult:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SecGemmaLeanRunnerError("git_check_unavailable") from exc
    return _GitResult(
        returncode=int(completed.returncode),
        stdout=bytes(completed.stdout),
    )


def _git_text(
    repo_root: Path,
    *arguments: str,
    git_runner: GitRunner = _default_git,
) -> str:
    result = git_runner(repo_root, tuple(arguments))
    if result.returncode != 0:
        raise SecGemmaLeanRunnerError("git_state_invalid")
    try:
        return result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise SecGemmaLeanRunnerError("git_state_invalid") from exc


def _git_success(
    repo_root: Path,
    *arguments: str,
    git_runner: GitRunner = _default_git,
) -> bool:
    return git_runner(repo_root, tuple(arguments)).returncode == 0


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_FLAG)


def _canonical_repo_root(value: Path) -> Path:
    if not isinstance(value, Path):
        raise SecGemmaLeanRunnerError("repo_root_invalid")
    try:
        absolute = value.absolute()
        details = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise SecGemmaLeanRunnerError("repo_root_invalid") from exc
    if (
        not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or resolved != absolute
        or not (resolved / ".git").is_dir()
    ):
        raise SecGemmaLeanRunnerError("repo_root_invalid")
    return resolved


def _canonical_relative_path(value: str, *, suffix: str | None = None) -> str:
    if type(value) is not str or "\\" in value:
        raise SecGemmaLeanRunnerError("path_contract_invalid")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != value
        or (suffix is not None and pure.suffix != suffix)
    ):
        raise SecGemmaLeanRunnerError("path_contract_invalid")
    return value


def _require_sha1(value: str) -> str:
    if type(value) is not str or _SHA1_RE.fullmatch(value) is None:
        raise SecGemmaLeanRunnerError("git_state_invalid")
    return value


def _read_safe_source(repo_root: Path, relative_path: str) -> bytes:
    path = repo_root
    for part in PurePosixPath(relative_path).parts:
        path = path / part
        try:
            details = path.lstat()
        except OSError as exc:
            raise SecGemmaLeanRunnerError("executed_source_invalid") from exc
        if stat.S_ISLNK(details.st_mode) or _is_reparse(details):
            raise SecGemmaLeanRunnerError("executed_source_invalid")
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise SecGemmaLeanRunnerError("executed_source_invalid")
    try:
        if path.resolve(strict=True) != path.absolute():
            raise SecGemmaLeanRunnerError("executed_source_invalid")
        return path.read_bytes()
    except OSError as exc:
        raise SecGemmaLeanRunnerError("executed_source_invalid") from exc


def _verify_execution_and_source_bytes(
    repo_root: Path,
    *,
    git_runner: GitRunner = _default_git,
) -> dict[str, Any]:
    root = _canonical_repo_root(repo_root)
    try:
        expected_runner = (root / Path(RUNNER_SOURCE_PATH)).resolve(strict=True)
        actual_runner = Path(__file__).resolve(strict=True)
    except OSError as exc:
        raise SecGemmaLeanRunnerError("execution_root_mismatch") from exc
    if actual_runner != expected_runner:
        raise SecGemmaLeanRunnerError("execution_root_mismatch")

    paths = tuple(sorted({RUNNER_SOURCE_PATH, *FROZEN_SCIENTIFIC_PATHS}))
    worktree_hashes: dict[str, str] = {}
    loaded_modules: list[str] = []
    for relative_path in paths:
        path = _canonical_relative_path(relative_path, suffix=".py")
        worktree_bytes = _read_safe_source(root, path)
        committed = git_runner(root, ("show", f"HEAD:{path}"))
        if committed.returncode != 0 or committed.stdout != worktree_bytes:
            raise SecGemmaLeanRunnerError("executed_source_invalid")
        worktree_hashes[path] = hashlib.sha256(worktree_bytes).hexdigest()

        module_name = (
            "agent_benchmark"
            if path == PACKAGE_INIT_SOURCE_PATH
            else path[:-3].replace("/", ".")
        )
        loaded = sys.modules.get(module_name)
        if loaded is None:
            continue
        module_file = getattr(loaded, "__file__", None)
        try:
            loaded_path = Path(module_file).resolve(strict=True)
        except (OSError, TypeError):
            raise SecGemmaLeanRunnerError("execution_root_mismatch") from None
        if loaded_path != (root / Path(path)).resolve(strict=True):
            raise SecGemmaLeanRunnerError("execution_root_mismatch")
        loaded_modules.append(module_name)

    return {
        "executing_repository_root_bound": True,
        "runner_source_path": RUNNER_SOURCE_PATH,
        "bound_source_count": len(worktree_hashes),
        "bound_worktree_source_sha256s": worktree_hashes,
        "bound_worktree_source_sha256s_sha256": canonical_sha256(
            worktree_hashes
        ),
        "loaded_source_modules_bound": sorted(loaded_modules),
    }


def _verify_dependency_identity(repo_root: Path) -> dict[str, Any]:
    from agent_benchmark import sec_gemma_online_risk_overlay_source_verifier as verifier

    root = _canonical_repo_root(repo_root)
    new_inventory = dict(NEW_SOURCE_FILES)
    participant_paths = {
        verifier.CONTRACT_SOURCE_PATH,
        *SOURCE_PIN_FILES.values(),
        *new_inventory.values(),
    }
    try:
        dependency_paths, external_imports = verifier._dependency_closure(
            root,
            participant_paths=participant_paths,
            new_inventory=new_inventory,
        )
        external_distributions = verifier._external_distribution_identities(
            external_imports
            & set(verifier._ALLOWED_EXTERNAL_IMPORT_DISTRIBUTIONS)
        )
        numerical_distributions = (
            verifier._numerical_time_distribution_identities(
                root,
                external_imports,
            )
        )
        dependency_sources = [
            {
                "path": path,
                "sha256": hashlib.sha256(
                    verifier._verify_real_source(root, path)
                ).hexdigest(),
            }
            for path in dependency_paths
        ]
    except Exception:
        raise SecGemmaLeanRunnerError("dependency_identity_invalid") from None

    dependency_material = {
        "dependency_sources": dependency_sources,
        "external_distributions": [
            {
                "import_name": import_name,
                "distribution": distribution,
                "version": version,
                "module_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in module_files
                ],
                "module_files_sha256": module_files_sha256,
                "direct_url_sha256": direct_url_sha256,
            }
            for (
                import_name,
                distribution,
                version,
                module_files,
                module_files_sha256,
                direct_url_sha256,
            ) in external_distributions
        ],
        "numerical_time_distributions": [
            {
                "import_name": import_name,
                "distribution": distribution,
                "version": version,
                "module_origin": {
                    "path": module_origin,
                    "sha256": module_origin_sha256,
                },
                "runtime_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in runtime_files
                ],
                "runtime_files_sha256": runtime_files_sha256,
                "direct_url_sha256": direct_url_sha256,
            }
            for (
                import_name,
                distribution,
                version,
                module_origin,
                module_origin_sha256,
                runtime_files,
                runtime_files_sha256,
                direct_url_sha256,
            ) in numerical_distributions
        ],
    }
    observed = canonical_sha256(dependency_material)
    if observed != V22_DEPENDENCY_CLOSURE_SHA256:
        raise SecGemmaLeanRunnerError("dependency_identity_invalid")
    return {
        "verified_v22_dependency_closure_sha256": observed,
        "dependency_source_count": len(dependency_sources),
        "external_distribution_count": len(external_distributions),
        "numerical_time_distribution_count": len(numerical_distributions),
        "dependency_identity_recomputed": True,
    }


def _verify_repository(
    repo_root: Path,
    *,
    git_runner: GitRunner = _default_git,
    source_tree_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
    dependency_verifier: Callable[[Path], Mapping[str, Any]] = (
        _verify_dependency_identity
    ),
) -> dict[str, Any]:
    root = _canonical_repo_root(repo_root)
    branch = _git_text(
        root,
        "branch",
        "--show-current",
        git_runner=git_runner,
    )
    if branch != BRANCH_NAME:
        raise SecGemmaLeanRunnerError("wrong_branch")

    status = git_runner(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    )
    if status.returncode != 0 or status.stdout:
        raise SecGemmaLeanRunnerError("working_tree_not_clean")

    origin_url = _git_text(
        root,
        "remote",
        "get-url",
        "origin",
        git_runner=git_runner,
    )
    if origin_url != EXPECTED_ORIGIN_URL:
        raise SecGemmaLeanRunnerError("origin_changed")

    head = _require_sha1(
        _git_text(root, "rev-parse", "HEAD", git_runner=git_runner)
    )
    upstream_name = _git_text(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        git_runner=git_runner,
    )
    if upstream_name != f"origin/{BRANCH_NAME}":
        raise SecGemmaLeanRunnerError("upstream_changed")
    upstream = _require_sha1(
        _git_text(
            root,
            "rev-parse",
            "@{upstream}",
            git_runner=git_runner,
        )
    )
    if head != upstream:
        raise SecGemmaLeanRunnerError("head_not_in_cached_origin")

    for ancestor in (PREREGISTRATION_COMMIT, SCIENTIFIC_PARENT_COMMIT):
        if not _git_success(
            root,
            "merge-base",
            "--is-ancestor",
            ancestor,
            head,
            git_runner=git_runner,
        ):
            raise SecGemmaLeanRunnerError("required_ancestor_missing")

    parent_tree = _require_sha1(
        _git_text(
            root,
            "rev-parse",
            f"{SCIENTIFIC_PARENT_COMMIT}^{{tree}}",
            git_runner=git_runner,
        )
    )
    if parent_tree != SCIENTIFIC_PARENT_TREE:
        raise SecGemmaLeanRunnerError("scientific_parent_changed")

    preregistration_head_blob = _require_sha1(
        _git_text(
            root,
            "rev-parse",
            f"HEAD:{PREREGISTRATION_PATH}",
            git_runner=git_runner,
        )
    )
    preregistration_frozen_blob = _require_sha1(
        _git_text(
            root,
            "rev-parse",
            f"{PREREGISTRATION_COMMIT}:{PREREGISTRATION_PATH}",
            git_runner=git_runner,
        )
    )
    if preregistration_head_blob != preregistration_frozen_blob:
        raise SecGemmaLeanRunnerError("preregistration_changed")

    source_blobs: dict[str, str] = {}
    for raw_path in FROZEN_SCIENTIFIC_PATHS:
        path = _canonical_relative_path(raw_path, suffix=".py")
        head_blob = _require_sha1(
            _git_text(
                root,
                "rev-parse",
                f"HEAD:{path}",
                git_runner=git_runner,
            )
        )
        parent_blob = _require_sha1(
            _git_text(
                root,
                "rev-parse",
                f"{SCIENTIFIC_PARENT_COMMIT}:{path}",
                git_runner=git_runner,
            )
        )
        if head_blob != parent_blob:
            raise SecGemmaLeanRunnerError("scientific_source_changed")
        source_blobs[path] = head_blob

    if source_tree_verifier is None:
        source_integrity = _verify_execution_and_source_bytes(
            root,
            git_runner=git_runner,
        )
    else:
        source_integrity = dict(source_tree_verifier(root))
    dependency_identity = dict(dependency_verifier(root))

    return {
        "branch": branch,
        "head_commit": head,
        "upstream_ref": upstream_name,
        "cached_upstream_commit": upstream,
        "remote_state_basis": "local_cached_origin_tracking_ref",
        "origin_url": origin_url,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_blob": preregistration_head_blob,
        "scientific_parent_commit": SCIENTIFIC_PARENT_COMMIT,
        "scientific_parent_tree": parent_tree,
        "declared_v22_preregistration_commit": V22_PREREGISTRATION_COMMIT,
        "declared_v22_contract_manifest_sha256": V22_CONTRACT_MANIFEST_SHA256,
        "declared_v22_implementation_manifest_sha256": (
            V22_IMPLEMENTATION_MANIFEST_SHA256
        ),
        "working_tree_clean_at_verification": True,
        "frozen_scientific_source_count": len(source_blobs),
        "frozen_scientific_source_blobs": source_blobs,
        "frozen_scientific_source_blobs_sha256": canonical_sha256(
            source_blobs
        ),
        "executed_source_identity": source_integrity,
        "dependency_identity": dependency_identity,
    }


def _require_owned_regular_file(path: Path, repo_root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        details = path.lstat()
        relative = resolved.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        raise SecGemmaLeanRunnerError("private_config_invalid") from exc
    if (
        not relative.parts
        or not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
        or resolved != path.absolute()
    ):
        raise SecGemmaLeanRunnerError("private_config_invalid")
    return resolved


def _read_private_config_bytes(config_path: Path) -> bytes | None:
    descriptor: int | None = None
    raw = b""
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(config_path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_PRIVATE_CONFIG_BYTES
        ):
            return None
        raw = os.read(descriptor, MAX_PRIVATE_CONFIG_BYTES + 1)
        if len(raw) > MAX_PRIVATE_CONFIG_BYTES or len(raw) != opened.st_size:
            raw = b""
            return None
        current = config_path.stat()
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            current.st_dev,
            current.st_ino,
            current.st_size,
        ):
            raw = b""
            return None
        return raw
    except OSError:
        raw = b""
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _fingerprint_private_config(raw: bytes) -> str | None:
    payload: Any = None
    private_contact: Any = None
    audit: Any = None
    fingerprint: str | None = None
    try:
        from agent_benchmark.sec_point_in_time import validate_sec_user_agent

        text = raw.decode("utf-8", errors="strict")
        payload = json.loads(text)
        text = ""
        if type(payload) is not dict or type(payload.get("secrets")) is not dict:
            raise ValueError("private config shape changed")
        private_contact = payload["secrets"].get("sec_user_agent")
        audit = validate_sec_user_agent(private_contact)
        if (
            audit.real_contact_validated is not True
            or _SHA256_TAGGED_RE.fullmatch(audit.sha256) is None
        ):
            raise ValueError("private contact audit changed")
        fingerprint = audit.sha256
    except Exception:
        fingerprint = None
    finally:
        raw = b""
        payload = None
        private_contact = None
        audit = None
    return fingerprint


def _contact_fingerprint(
    repo_root: Path,
    *,
    git_runner: GitRunner = _default_git,
) -> str:
    root = _canonical_repo_root(repo_root)
    relative = _canonical_relative_path(PRIVATE_CONFIG_PATH, suffix=".json")
    if _git_success(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        relative,
        git_runner=git_runner,
    ):
        raise SecGemmaLeanRunnerError("private_config_tracked")
    if not _git_success(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        relative,
        git_runner=git_runner,
    ):
        raise SecGemmaLeanRunnerError("private_config_not_ignored")
    _safe_directory((root / Path(relative)).parent, root)
    config_path = _require_owned_regular_file(root / Path(relative), root)
    raw = _read_private_config_bytes(config_path)
    if raw is None:
        raise SecGemmaLeanRunnerError("private_contact_invalid") from None
    fingerprint = _fingerprint_private_config(raw)
    raw = b""
    if (
        type(fingerprint) is not str
        or _SHA256_TAGGED_RE.fullmatch(fingerprint) is None
    ):
        del fingerprint
        raise SecGemmaLeanRunnerError("private_contact_invalid")
    return fingerprint


def _safe_directory(path: Path, repo_root: Path) -> None:
    try:
        relative = path.absolute().relative_to(repo_root)
    except ValueError as exc:
        raise SecGemmaLeanRunnerError("checkpoint_root_invalid") from exc
    current = repo_root
    for part in relative.parts:
        current = current / part
        if not current.exists():
            continue
        try:
            details = current.lstat()
        except OSError as exc:
            raise SecGemmaLeanRunnerError("checkpoint_root_invalid") from exc
        if stat.S_ISLNK(details.st_mode) or _is_reparse(details):
            raise SecGemmaLeanRunnerError("checkpoint_root_invalid")
        if current != path and not stat.S_ISDIR(details.st_mode):
            raise SecGemmaLeanRunnerError("checkpoint_root_invalid")


def _prepare_checkpoint_root(
    repo_root: Path,
    *,
    git_runner: GitRunner = _default_git,
) -> Path:
    root = _canonical_repo_root(repo_root)
    relative = _canonical_relative_path(CHECKPOINT_ROOT)
    ignored_probe = f"{relative}/.ignore-probe"
    if not _git_success(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        ignored_probe,
        git_runner=git_runner,
    ):
        raise SecGemmaLeanRunnerError("checkpoint_root_not_ignored")
    checkpoint = root / Path(relative)
    _safe_directory(checkpoint, root)
    try:
        checkpoint.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SecGemmaLeanRunnerError("checkpoint_root_unwritable") from exc
    _safe_directory(checkpoint, root)
    if not checkpoint.is_dir():
        raise SecGemmaLeanRunnerError("checkpoint_root_invalid")
    return checkpoint


class _RunLease(AbstractContextManager["_RunLease"]):
    """Cooperative preflight-only marker, held for every preflight check.

    This is intentionally not execution authority or crash-recovery state.  A
    later acquisition command must reacquire its own lease and rerun readiness.
    """

    def __init__(self, path: Path):
        self._path = path
        self._handle: Any | None = None
        self._nonce = uuid.uuid4().hex
        self._identity: tuple[int, int] | None = None
        self._payload_sha256: str | None = None

    def _owned_path_is_present(self, *, require_complete_payload: bool) -> bool:
        if self._identity is None:
            return False
        try:
            details = self._path.lstat()
            if (
                not stat.S_ISREG(details.st_mode)
                or stat.S_ISLNK(details.st_mode)
                or _is_reparse(details)
                or details.st_nlink != 1
                or (details.st_dev, details.st_ino) != self._identity
            ):
                return False
            if require_complete_payload:
                return (
                    self._payload_sha256 is not None
                    and hashlib.sha256(self._path.read_bytes()).hexdigest()
                    == self._payload_sha256
                )
            return True
        except OSError:
            return False

    def __enter__(self) -> "_RunLease":
        payload = json.dumps(
            {
                "schema_version": "sec-gemma-lean-v3-local-lease-v1",
                "pid": os.getpid(),
                "nonce_sha256": hashlib.sha256(
                    self._nonce.encode("ascii")
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        created = False
        try:
            self._handle = self._path.open("xb")
            created = True
            opened = os.fstat(self._handle.fileno())
            self._identity = (opened.st_dev, opened.st_ino)
            self._payload_sha256 = hashlib.sha256(payload).hexdigest()
            self._handle.write(payload)
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except FileExistsError as exc:
            raise SecGemmaLeanRunnerError("active_run_present") from exc
        except OSError as exc:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None
            if created:
                if self._owned_path_is_present(require_complete_payload=False):
                    try:
                        self._path.unlink(missing_ok=True)
                    except OSError:
                        pass
            raise SecGemmaLeanRunnerError("run_lock_unavailable") from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        close_failed = False
        unlink_failed = False
        owned_before_close = self._owned_path_is_present(
            require_complete_payload=True
        )
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                close_failed = True
            finally:
                self._handle = None
        owned_after_close = self._owned_path_is_present(
            require_complete_payload=True
        )
        if owned_before_close and owned_after_close:
            try:
                self._path.unlink()
            except OSError:
                unlink_failed = True
        else:
            unlink_failed = True
        if close_failed or unlink_failed:
            raise SecGemmaLeanRunnerError("run_lock_cleanup_failed") from None
        return False


def _write_probe(checkpoint_root: Path) -> None:
    probe = checkpoint_root / f".write-probe-{uuid.uuid4().hex}"
    try:
        with probe.open("xb") as handle:
            handle.write(b"sec-gemma-lean-v3-write-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
        if probe.read_bytes() != b"sec-gemma-lean-v3-write-probe\n":
            raise SecGemmaLeanRunnerError("checkpoint_probe_mismatch")
    except SecGemmaLeanRunnerError:
        raise
    except OSError as exc:
        raise SecGemmaLeanRunnerError("checkpoint_root_unwritable") from exc
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError as exc:
            raise SecGemmaLeanRunnerError("checkpoint_probe_cleanup_failed") from exc


class _RestrictedRuntimeProbeTransport:
    """Record and permit only one exact sequence of loopback identity calls."""

    def __init__(
        self,
        delegate: Any,
        allowed_requests: tuple[tuple[str, str], ...],
    ):
        if not callable(getattr(delegate, "request", None)):
            raise SecGemmaLeanRunnerError("runtime_transport_invalid")
        if (
            not allowed_requests
            or any(
                type(method) is not str
                or type(endpoint) is not str
                or not method
                or not endpoint.startswith("http://127.0.0.1:")
                for method, endpoint in allowed_requests
            )
        ):
            raise SecGemmaLeanRunnerError("runtime_transport_invalid")
        self._delegate = delegate
        self._allowed_requests = allowed_requests
        self._calls: list[tuple[str, str]] = []
        self._closed = False

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._calls)

    def request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        position = len(self._calls)
        if (
            self._closed
            or position >= len(self._allowed_requests)
            or (method, endpoint) != self._allowed_requests[position]
        ):
            raise SecGemmaLeanRunnerError("runtime_probe_request_forbidden")
        self._calls.append((method, endpoint))
        return self._delegate.request(method, endpoint, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()


def _load_verified_runtime_modules() -> tuple[Any, Any, Any]:
    from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
        load_allowed_requests,
    )

    verified_requests = load_allowed_requests()
    from agent_benchmark import sec_filing_gemma_ollama as ollama
    from agent_benchmark import sec_gemma_online_risk_overlay_production as production

    if (
        sys.modules.get("requests") is not verified_requests
        or getattr(ollama, "requests", None) is not verified_requests
        or getattr(production, "requests", None) is not verified_requests
    ):
        raise SecGemmaLeanRunnerError("runtime_transport_invalid")
    return verified_requests, ollama, production


def _load_production_runtime_probe(
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    verified_requests, ollama, production = _load_verified_runtime_modules()

    allowed = (
        ("GET", ollama.OLLAMA_VERSION_ENDPOINT),
        ("POST", ollama.OLLAMA_SHOW_ENDPOINT),
    )
    delegate = ollama.build_hardened_loopback_session()
    if type(delegate) is not verified_requests.Session:
        try:
            delegate.close()
        except Exception:
            pass
        raise SecGemmaLeanRunnerError("runtime_transport_invalid")
    transport = _RestrictedRuntimeProbeTransport(delegate, allowed)
    probe_failed = False
    close_failed = False
    payload: dict[str, Any] = {}
    try:
        payload = dict(
            production._runtime_identity_payload(transport=transport)
        )
    except Exception:
        probe_failed = True
    finally:
        try:
            transport.close()
        except Exception:
            close_failed = True
    if probe_failed or close_failed or transport.calls != allowed:
        raise SecGemmaLeanRunnerError("runtime_identity_invalid") from None
    return payload, transport.calls


def _verify_runtime(
    *,
    probe_loader: Callable[[], Mapping[str, Any]] | None = None,
    bundle_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    test_injections_used = probe_loader is not None or bundle_verifier is not None
    if bundle_verifier is None:
        from agent_benchmark.sec_gemma_online_risk_overlay_runtime import (
            verify_installed_pinned_runtime,
        )

        bundle_verifier = verify_installed_pinned_runtime
    try:
        if probe_loader is None:
            payload, observed_calls = _load_production_runtime_probe()
        else:
            payload = dict(probe_loader())
            observed_calls = ()
        if set(payload) != {"version_response_hex", "show_response_hex"}:
            raise ValueError("runtime probe keys changed")
        version_bytes = bytes.fromhex(payload["version_response_hex"])
        show_bytes = bytes.fromhex(payload["show_response_hex"])
        receipt = dict(
            bundle_verifier(
                version_response_bytes=version_bytes,
                show_response_bytes=show_bytes,
            )
        )
    except Exception:
        raise SecGemmaLeanRunnerError("runtime_identity_invalid") from None
    if (
        receipt.get("model_name") != MODEL_NAME
        or receipt.get("runtime_fingerprint_sha256")
        != RUNTIME_FINGERPRINT_SHA256
        or type(receipt.get("runtime_receipt_sha256")) is not str
    ):
        raise SecGemmaLeanRunnerError("runtime_identity_invalid")
    trusted_counts: dict[str, int | None]
    if test_injections_used:
        trusted_counts = {
            "external_network_requests": None,
            "loopback_identity_requests": None,
            "model_generation_calls": None,
        }
    else:
        trusted_counts = {
            "external_network_requests": 0,
            "loopback_identity_requests": len(observed_calls),
            "model_generation_calls": 0,
        }
    return {
        "model_name": MODEL_NAME,
        "runtime_fingerprint_sha256": RUNTIME_FINGERPRINT_SHA256,
        "runtime_receipt_sha256": receipt["runtime_receipt_sha256"],
        "model_manifest_sha256": receipt.get("manifest_sha256"),
        "model_config_sha256": receipt.get("config_sha256"),
        "ordered_layer_content_sha256s": receipt.get(
            "ordered_layer_content_sha256s"
        ),
        "ollama_version_response_sha256": receipt.get(
            "version_response_sha256"
        ),
        "ollama_show_semantic_sha256": receipt.get(
            "show_response_semantic_sha256"
        ),
        "ollama_show_raw_sha256_diagnostic": receipt.get(
            "show_response_raw_sha256"
        ),
        "identity_probe": {
            "restricted_loopback_sequence_enforced": not test_injections_used,
            "observed_request_count": (
                len(observed_calls) if not test_injections_used else None
            ),
            "test_injections_used": test_injections_used,
        },
        "observed_effect_counts": trusted_counts,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _expected_runtime_effect_counts() -> dict[str, int]:
    return {
        "external_network_requests": 0,
        "loopback_identity_requests": 2,
        "model_generation_calls": 0,
    }


def _expected_preflight_effect_counts() -> dict[str, int]:
    return {
        **_expected_runtime_effect_counts(),
        "official_sec_requests": 0,
        "market_data_requests": 0,
        "paid_api_calls": 0,
        "trading_actions": 0,
        "performance_results_opened": 0,
    }


def _validated_sealable_report(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaLeanRunnerError("untrusted_report_not_sealable")
    report = value
    expected_keys = {
        "schema_version",
        "status",
        "created_at_utc",
        "elapsed_seconds_hex",
        "repository",
        "repository_observation_timing",
        "private_sec_contact",
        "local_runtime",
        "checkpoint",
        "effect_counts",
        "effect_counts_basis",
        "trust",
        "execution_authorized",
        "next_step",
        "preflight_sha256",
    }
    repository = report.get("repository")
    runtime = report.get("local_runtime")
    contact = report.get("private_sec_contact")
    checkpoint = report.get("checkpoint")
    dependency_identity = (
        repository.get("dependency_identity", {})
        if type(repository) is dict
        else {}
    )
    source_identity = (
        repository.get("executed_source_identity", {})
        if type(repository) is dict
        else {}
    )
    if type(dependency_identity) is not dict:
        dependency_identity = {}
    if type(source_identity) is not dict:
        source_identity = {}
    loaded_modules = source_identity.get("loaded_source_modules_bound", [])
    required_loaded_modules = {
        "agent_benchmark.sec_gemma_online_risk_overlay_production",
        "agent_benchmark.sec_gemma_online_risk_overlay_runtime",
        "agent_benchmark.sec_gemma_online_risk_overlay_source_verifier",
        "agent_benchmark.sec_point_in_time",
    }
    try:
        elapsed = float.fromhex(report.get("elapsed_seconds_hex", ""))
    except (TypeError, ValueError):
        elapsed = -1.0
    if (
        set(report) != expected_keys
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("effect_counts") != _expected_preflight_effect_counts()
        or report.get("effect_counts_basis")
        != {
            "trusted": True,
            "runtime_requests_recorded_by_restricted_transport": True,
            "other_preflight_effects_bound_by_fixed_code_path": True,
        }
        or report.get("trust")
        != {
            "production_defaults_only": True,
            "test_injections_used": False,
            "eligible_for_public_seal": True,
        }
        or report.get("execution_authorized") is not False
        or report.get("next_step")
        != "development_source_acquisition_preflight"
        or not math.isfinite(elapsed)
        or elapsed < 0.0
        or type(report.get("created_at_utc")) is not str
        or _UTC_TIMESTAMP_RE.fullmatch(report["created_at_utc"]) is None
        or report.get("repository_observation_timing")
        != {
            "cleanliness_observed_before_artifact_write": True,
            "post_artifact_cleanliness_claimed": False,
            "artifact_write_will_dirty_worktree_until_commit": True,
        }
        or type(repository) is not dict
        or repository.get("branch") != BRANCH_NAME
        or repository.get("working_tree_clean_at_verification") is not True
        or repository.get("head_commit")
        != repository.get("cached_upstream_commit")
        or _SHA1_RE.fullmatch(repository.get("head_commit", "")) is None
        or repository.get("reverified_after_runtime") is not True
        or repository.get("pre_runtime_stable_repository_sha256")
        != repository.get("post_runtime_stable_repository_sha256")
        or _SHA256_RE.fullmatch(
            repository.get("pre_runtime_stable_repository_sha256", "")
        )
        is None
        or dependency_identity.get("verified_v22_dependency_closure_sha256")
        != V22_DEPENDENCY_CLOSURE_SHA256
        or dependency_identity.get("dependency_identity_recomputed") is not True
        or dependency_identity.get("dependency_source_count") != 1
        or dependency_identity.get("external_distribution_count") != 5
        or dependency_identity.get("numerical_time_distribution_count") != 2
        or source_identity.get("executing_repository_root_bound") is not True
        or source_identity.get("runner_source_path") != RUNNER_SOURCE_PATH
        or source_identity.get("bound_source_count")
        != len({RUNNER_SOURCE_PATH, *FROZEN_SCIENTIFIC_PATHS})
        or type(loaded_modules) is not list
        or not required_loaded_modules.issubset(set(loaded_modules))
        or type(runtime) is not dict
        or runtime.get("model_name") != MODEL_NAME
        or runtime.get("runtime_fingerprint_sha256")
        != RUNTIME_FINGERPRINT_SHA256
        or runtime.get("observed_effect_counts")
        != _expected_runtime_effect_counts()
        or _SHA256_RE.fullmatch(runtime.get("runtime_receipt_sha256", ""))
        is None
        or runtime.get("identity_probe")
        != {
            "restricted_loopback_sequence_enforced": True,
            "observed_request_count": 2,
            "test_injections_used": False,
        }
        or type(contact) is not dict
        or _SHA256_TAGGED_RE.fullmatch(contact.get("sha256", "")) is None
        or contact.get("plausible_format_validated") is not True
        or contact.get("reachability_not_proven") is not True
        or contact.get("readable_value_published") is not False
        or checkpoint
        != {
            "root": CHECKPOINT_ROOT,
            "ignored_local_state": True,
            "exclusive_lock_held_during_checks": True,
            "preflight_only_cooperative_marker": True,
            "crash_can_leave_stale_marker": True,
            "lease_released_after_successful_preflight": True,
            "acquisition_must_reacquire_and_reverify": True,
            "write_flush_read_delete_probe_passed": True,
            "probe_file_retained": False,
        }
    ):
        raise SecGemmaLeanRunnerError("untrusted_report_not_sealable")
    body = {key: item for key, item in report.items() if key != "preflight_sha256"}
    try:
        valid_hash = canonical_sha256(body) == report["preflight_sha256"]
    except Exception:
        valid_hash = False
    if not valid_hash:
        raise SecGemmaLeanRunnerError("untrusted_report_not_sealable")
    return report


def _atomic_publish_preflight_report(
    repo_root: Path,
    report: dict[str, Any],
    checkpoint_root: Path,
) -> str:
    report = _validated_sealable_report(report)
    relative = _canonical_relative_path(PREFLIGHT_ARTIFACT_PATH, suffix=".json")
    target = repo_root / Path(relative)
    _safe_directory(target.parent, repo_root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SecGemmaLeanRunnerError("preflight_artifact_unwritable") from exc
    _safe_directory(target.parent, repo_root)
    if target.exists():
        raise SecGemmaLeanRunnerError("preflight_already_sealed")
    expected_checkpoint = repo_root / Path(CHECKPOINT_ROOT)
    _safe_directory(checkpoint_root, repo_root)
    if (
        checkpoint_root.absolute() != expected_checkpoint.absolute()
        or not checkpoint_root.is_dir()
    ):
        raise SecGemmaLeanRunnerError("checkpoint_root_invalid")
    payload = (
        json.dumps(
            dict(report),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    temporary = checkpoint_root / f".preflight-seal-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise SecGemmaLeanRunnerError("preflight_artifact_unwritable") from None
    try:
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
    except FileExistsError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise SecGemmaLeanRunnerError("preflight_already_sealed") from None
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise SecGemmaLeanRunnerError("preflight_artifact_unwritable") from None
    try:
        published = target.lstat()
        if (
            not stat.S_ISREG(published.st_mode)
            or stat.S_ISLNK(published.st_mode)
            or _is_reparse(published)
            or target.read_bytes() != payload
        ):
            raise OSError("published preflight differs from staged bytes")
        if os.name != "nt":
            temporary.unlink()
        final = target.lstat()
        if final.st_nlink != 1:
            raise OSError("published preflight retained another hard link")
    except OSError:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise SecGemmaLeanRunnerError("preflight_artifact_unwritable") from None
    return relative


def _run_preflight_core(
    repo_root: Path,
    *,
    repository_verifier: Callable[[Path], Mapping[str, Any]],
    contact_verifier: Callable[[Path], str],
    checkpoint_preparer: Callable[[Path], Path],
    runtime_verifier: Callable[[], Mapping[str, Any]],
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Shared core; injected checks are permanently unsealable test evidence."""

    root = _canonical_repo_root(repo_root)
    try:
        started = float(clock())
    except Exception as exc:
        raise SecGemmaLeanRunnerError("monotonic_clock_invalid") from exc
    if not math.isfinite(started):
        raise SecGemmaLeanRunnerError("monotonic_clock_invalid")

    checkpoint_root = checkpoint_preparer(root)
    lock_path = checkpoint_root / ACTIVE_LOCK_NAME
    with _RunLease(lock_path):
        repository_before = dict(repository_verifier(root))
        contact_sha256 = contact_verifier(root)
        if (
            type(contact_sha256) is not str
            or _SHA256_TAGGED_RE.fullmatch(contact_sha256) is None
        ):
            raise SecGemmaLeanRunnerError("private_contact_invalid")
        runtime = dict(runtime_verifier())
        repository_after = dict(repository_verifier(root))
        before_identity = dict(
            repository_before.get("executed_source_identity", {})
        )
        after_identity = dict(
            repository_after.get("executed_source_identity", {})
        )
        before_loaded = set(
            before_identity.pop("loaded_source_modules_bound", [])
        )
        after_loaded = set(
            after_identity.pop("loaded_source_modules_bound", [])
        )
        before_stable = {
            **repository_before,
            "executed_source_identity": before_identity,
        }
        after_stable = {
            **repository_after,
            "executed_source_identity": after_identity,
        }
        before_stable_sha256 = canonical_sha256(before_stable)
        after_stable_sha256 = canonical_sha256(after_stable)
        if (
            before_stable_sha256 != after_stable_sha256
            or not before_loaded.issubset(after_loaded)
        ):
            raise SecGemmaLeanRunnerError("repository_changed_during_preflight")
        repository = {
            **repository_after,
            "reverified_after_runtime": True,
            "pre_runtime_stable_repository_sha256": before_stable_sha256,
            "post_runtime_stable_repository_sha256": after_stable_sha256,
            "loaded_source_modules_bound_before_runtime": sorted(before_loaded),
        }
        effect_counts: dict[str, int | None] = {
            "external_network_requests": None,
            "loopback_identity_requests": None,
            "model_generation_calls": None,
            "official_sec_requests": None,
            "market_data_requests": None,
            "paid_api_calls": None,
            "trading_actions": None,
            "performance_results_opened": None,
        }
        _write_probe(checkpoint_root)
        try:
            finished = float(clock())
        except Exception as exc:
            raise SecGemmaLeanRunnerError("monotonic_clock_invalid") from exc
        elapsed = finished - started
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise SecGemmaLeanRunnerError("monotonic_clock_invalid")
        try:
            created_at = now()
        except Exception:
            raise SecGemmaLeanRunnerError("wall_clock_invalid") from None
        try:
            parsed_created_at = datetime.fromisoformat(
                created_at[:-1] + "+00:00"
            )
        except (TypeError, ValueError):
            parsed_created_at = None
        if (
            type(created_at) is not str
            or _UTC_TIMESTAMP_RE.fullmatch(created_at) is None
            or parsed_created_at is None
            or parsed_created_at.tzinfo != timezone.utc
        ):
            raise SecGemmaLeanRunnerError("wall_clock_invalid")
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "test_only_untrusted",
            "created_at_utc": created_at,
            "elapsed_seconds_hex": elapsed.hex(),
            "repository": repository,
            "repository_observation_timing": {
                "cleanliness_observed_before_artifact_write": True,
                "post_artifact_cleanliness_claimed": False,
                "artifact_write_will_dirty_worktree_until_commit": False,
            },
            "private_sec_contact": {
                "sha256": contact_sha256,
                "plausible_format_validated": True,
                "reachability_not_proven": True,
                "readable_value_published": False,
            },
            "local_runtime": runtime,
            "checkpoint": {
                "root": CHECKPOINT_ROOT,
                "ignored_local_state": True,
                "exclusive_lock_held_during_checks": True,
                "preflight_only_cooperative_marker": True,
                "crash_can_leave_stale_marker": True,
                "lease_released_after_successful_preflight": True,
                "acquisition_must_reacquire_and_reverify": True,
                "write_flush_read_delete_probe_passed": True,
                "probe_file_retained": False,
            },
            "effect_counts": effect_counts,
            "effect_counts_basis": {
                "trusted": False,
                "runtime_requests_recorded_by_restricted_transport": False,
                "other_preflight_effects_bound_by_fixed_code_path": False,
            },
            "trust": {
                "production_defaults_only": False,
                "test_injections_used": True,
                "eligible_for_public_seal": False,
            },
            "execution_authorized": False,
            "next_step": "development_source_acquisition_preflight",
        }
        report = {**body, "preflight_sha256": canonical_sha256(body)}
    return report


def _run_untrusted_preflight(
    repo_root: Path,
    *,
    repository_verifier: Callable[[Path], Mapping[str, Any]],
    contact_verifier: Callable[[Path], str],
    checkpoint_preparer: Callable[[Path], Path],
    runtime_verifier: Callable[[], Mapping[str, Any]],
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    """Run injected offline checks without any canonical sealing capability."""

    return _run_preflight_core(
        repo_root,
        repository_verifier=repository_verifier,
        contact_verifier=contact_verifier,
        checkpoint_preparer=checkpoint_preparer,
        runtime_verifier=runtime_verifier,
        clock=clock,
        now=now,
    )


def run_preflight(repo_root: Path, *, seal: bool = True) -> dict[str, Any]:
    """Run the fixed production preflight and optionally seal its receipt."""

    if type(seal) is not bool:
        raise SecGemmaLeanRunnerError("seal_mode_invalid")
    root = _canonical_repo_root(repo_root)
    if seal and (root / Path(PREFLIGHT_ARTIFACT_PATH)).exists():
        raise SecGemmaLeanRunnerError("preflight_already_sealed")
    untrusted = _run_preflight_core(
        root,
        repository_verifier=_verify_repository,
        contact_verifier=_contact_fingerprint,
        checkpoint_preparer=_prepare_checkpoint_root,
        runtime_verifier=_verify_runtime,
        clock=time.monotonic,
        now=_utc_now,
    )
    runtime_counts = untrusted.get("local_runtime", {}).get(
        "observed_effect_counts"
    )
    if runtime_counts != _expected_runtime_effect_counts():
        raise SecGemmaLeanRunnerError("runtime_effect_counts_invalid")
    body = {
        key: value
        for key, value in untrusted.items()
        if key != "preflight_sha256"
    }
    body.update(
        {
            "status": "passed",
            "repository_observation_timing": {
                "cleanliness_observed_before_artifact_write": True,
                "post_artifact_cleanliness_claimed": False,
                "artifact_write_will_dirty_worktree_until_commit": seal,
            },
            "effect_counts": _expected_preflight_effect_counts(),
            "effect_counts_basis": {
                "trusted": True,
                "runtime_requests_recorded_by_restricted_transport": True,
                "other_preflight_effects_bound_by_fixed_code_path": True,
            },
            "trust": {
                "production_defaults_only": True,
                "test_injections_used": False,
                "eligible_for_public_seal": True,
            },
        }
    )
    report = {**body, "preflight_sha256": canonical_sha256(body)}
    if seal:
        _atomic_publish_preflight_report(
            root,
            report,
            root / Path(CHECKPOINT_ROOT),
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lean SEC/Gemma v3 staged evidence runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight",
        help="run the zero-effect local readiness check",
    )
    preflight.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    arguments = parser.parse_args(argv)
    if arguments.command != "preflight":
        parser.error("unsupported command")
    try:
        report = run_preflight(arguments.repo_root, seal=True)
    except SecGemmaLeanRunnerError as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "reason_code": exc.code,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    except Exception:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "reason_code": "unexpected_preflight_failure",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 3
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "passed",
                "preflight_sha256": report["preflight_sha256"],
                "artifact": PREFLIGHT_ARTIFACT_PATH,
                "execution_authorized": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_LOCK_NAME",
    "BRANCH_NAME",
    "CHECKPOINT_ROOT",
    "FROZEN_SCIENTIFIC_PATHS",
    "PREFLIGHT_ARTIFACT_PATH",
    "PREREGISTRATION_COMMIT",
    "SCHEMA_VERSION",
    "SCIENTIFIC_PARENT_COMMIT",
    "SCIENTIFIC_PARENT_TREE",
    "SecGemmaLeanRunnerError",
    "main",
    "run_preflight",
]
