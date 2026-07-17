"""Zero-effect production preflight for SEC/Gemma lean evidence v3.2.

The only network-shaped operations in this module are two identity reads from
the loopback Ollama service.  Their order is fixed: ``GET /api/tags`` followed
by ``POST /api/show``.  The preflight has no SEC, Yahoo, market, generation,
performance, trading, paid-service, or external-network capability.

The public receipt is useful only when it is made by :func:`run_preflight`.
Injected helpers are intentionally routed through a test-only path which can
never be sealed as production evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Final
import uuid

from agent_benchmark import sec_gemma_lean_v32_delta as delta


SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-lean-evidence-v3-2-preflight-v1"
BRANCH_NAME: Final[str] = "codex/aapl-sec-gemma-lean-evidence-v3-2"
EXPECTED_UPSTREAM: Final[str] = f"origin/{BRANCH_NAME}"
EXPECTED_ORIGIN_URL: Final[str] = (
    "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git"
)
PRIVATE_CONFIG_PATH: Final[str] = "data/local_config.json"
CHECKPOINT_ROOT: Final[str] = "data/aapl_sec_gemma_lean_evidence_v3_2"
PREFLIGHT_ARTIFACT_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_evidence_v3_2/PREFLIGHT.json"
)
ACTIVE_LOCK_NAME: Final[str] = "ACTIVE.lock"
GLOBAL_DISPATCH_LOCK_NAME: Final[str] = "global_dispatch.lock"
RUN_LOCK_NAME: Final[str] = ".run.lock"
PROBE_INTENT_NAME: Final[str] = "IDENTITY_PROBE_INTENT.json"
PROBE_COMPLETION_NAME: Final[str] = "IDENTITY_PROBE_COMPLETE.json"
TEST_PROBE_INTENT_NAME: Final[str] = "TEST_IDENTITY_PROBE_INTENT.json"
TEST_PROBE_COMPLETION_NAME: Final[str] = "TEST_IDENTITY_PROBE_COMPLETE.json"
PUBLISH_COMPLETE_NAME: Final[str] = "PREFLIGHT_PUBLISH_COMPLETE.json"
MAX_PRIVATE_CONFIG_BYTES: Final[int] = 1024 * 1024
MAX_SOURCE_BYTES: Final[int] = 16 * 1024 * 1024

MODEL_NAME: Final[str] = "gemma4:12b"
MODEL_MANIFEST_SHA256: Final[str] = (
    "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
)
MODEL_CONFIG_DIGEST: Final[str] = (
    "c805f5b265d8e695c44f4065dfc368206cd8026447604925fef8db57ee32ee23"
)
MODEL_LAYER_DIGESTS: Final[tuple[str, ...]] = (
    "1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606",
    "675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842",
    "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594",
    "56380ca2ab89f1f68c283f4d50863c0bcab52ae3f1b9a88e4ab5617b176f71a3",
)
OLLAMA_VERSION: Final[str] = "0.32.0"
OLLAMA_VERSION_CANONICAL_BYTES: Final[bytes] = b'{"version":"0.32.0"}'
OLLAMA_VERSION_RESPONSE_SHA256: Final[str] = (
    "2bd89ec9b983123a225f3df0381c737a45302bb7417e345bf9ef92304e4388cf"
)
OLLAMA_SHOW_SEMANTIC_SHA256: Final[str] = (
    "5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6"
)
MODEL_INFO_SHA256: Final[str] = (
    "d21c1c125758901fcea224a7cb9df1057aeba7ebb5177b82d6ba1a096d65fc7b"
)
RUNTIME_FINGERPRINT_SHA256: Final[str] = (
    "816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77"
)

LOOPBACK_HOST: Final[str] = "127.0.0.1"
LOOPBACK_PORT: Final[int] = 11434
TAGS_PATH: Final[str] = "/api/tags"
SHOW_PATH: Final[str] = "/api/show"
ALLOWED_IDENTITY_SEQUENCE: Final[tuple[tuple[str, str], ...]] = (
    ("GET", TAGS_PATH),
    ("POST", SHOW_PATH),
)
SHOW_REQUEST_BYTES: Final[bytes] = b'{"model":"gemma4:12b","verbose":false}'
MAX_IDENTITY_RESPONSE_BYTES: Final[int] = 4 * 1024 * 1024
HTTP_TIMEOUT_SECONDS: Final[float] = 15.0

RUNNER_SOURCE_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_lean_v32_preflight.py"
)
EXECUTED_SOURCE_CLOSURE: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "agent_benchmark/__init__.py",
            RUNNER_SOURCE_PATH,
            "agent_benchmark/sec_gemma_lean_v32_delta.py",
            "agent_benchmark/sec_point_in_time.py",
            "agent_benchmark/sec_gemma_online_risk_overlay_runtime.py",
            "agent_benchmark/sec_gemma_online_risk_overlay_contract.py",
            "agent_benchmark/sec_filing_gemma_extractor_prompt.py",
            "agent_benchmark/sec_filing_gemma_extractor_schema.py",
        }
    )
)

_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z\Z"
)
_REPARSE_FLAG: Final[int] = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class SecGemmaLeanV32PreflightError(RuntimeError):
    """A fixed, public-safe preflight check failed."""

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
        return f"SecGemmaLeanV32PreflightError(code={self.code!r})"


@dataclass(frozen=True)
class _GitResult:
    returncode: int
    stdout: bytes


GitRunner = Callable[[Path, tuple[str, ...]], _GitResult]


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_FLAG)


def _resolved_git_executable() -> str:
    candidate = shutil.which("git")
    if candidate is None:
        raise SecGemmaLeanV32PreflightError("git_check_unavailable")
    try:
        path = Path(candidate).absolute()
        details = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError("git_check_unavailable") from exc
    if (
        not path.is_absolute()
        or path != resolved
        or not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
    ):
        raise SecGemmaLeanV32PreflightError("git_check_unavailable")
    return str(path)


_GIT_EXECUTABLE: Final[str] = _resolved_git_executable()


def _git_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _default_git(repo_root: Path, arguments: tuple[str, ...]) -> _GitResult:
    try:
        result = subprocess.run(
            [_GIT_EXECUTABLE, *arguments],
            cwd=repo_root,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SecGemmaLeanV32PreflightError("git_check_unavailable") from exc
    return _GitResult(int(result.returncode), bytes(result.stdout))


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SecGemmaLeanV32PreflightError("receipt_not_canonical") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_json_object(payload: bytes, code: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload or payload.startswith(b"\xef\xbb\xbf"):
        raise SecGemmaLeanV32PreflightError(code)

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise SecGemmaLeanV32PreflightError(code)
            result[key] = value
        return result

    def reject_constant(_: str) -> Any:
        raise SecGemmaLeanV32PreflightError(code)

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecGemmaLeanV32PreflightError(code) from exc
    if type(value) is not dict:
        raise SecGemmaLeanV32PreflightError(code)
    return value


def _canonical_repo_root(value: Path) -> Path:
    if not isinstance(value, Path):
        raise SecGemmaLeanV32PreflightError("repo_root_invalid")
    try:
        absolute = value.absolute()
        resolved = absolute.resolve(strict=True)
        details = absolute.lstat()
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError("repo_root_invalid") from exc
    git_marker = resolved / ".git"
    if (
        resolved != absolute
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or not git_marker.exists()
    ):
        raise SecGemmaLeanV32PreflightError("repo_root_invalid")
    return resolved


def _relative_path(value: str, *, suffix: str | None = None) -> str:
    if type(value) is not str or "\\" in value:
        raise SecGemmaLeanV32PreflightError("path_contract_invalid")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != value
        or (suffix is not None and pure.suffix != suffix)
    ):
        raise SecGemmaLeanV32PreflightError("path_contract_invalid")
    return value


def _git_text(
    repo_root: Path,
    *arguments: str,
    git_runner: GitRunner = _default_git,
) -> str:
    result = git_runner(repo_root, tuple(arguments))
    if result.returncode != 0:
        raise SecGemmaLeanV32PreflightError("git_state_invalid")
    try:
        return result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise SecGemmaLeanV32PreflightError("git_state_invalid") from exc


def _git_success(
    repo_root: Path,
    *arguments: str,
    git_runner: GitRunner = _default_git,
) -> bool:
    return git_runner(repo_root, tuple(arguments)).returncode == 0


def _sha1(value: str) -> str:
    if type(value) is not str or _SHA1_RE.fullmatch(value) is None:
        raise SecGemmaLeanV32PreflightError("git_state_invalid")
    return value


def _assert_git_toplevel(repo_root: Path, *, git_runner: GitRunner) -> None:
    raw = _git_text(
        repo_root,
        "rev-parse",
        "--show-toplevel",
        git_runner=git_runner,
    )
    try:
        observed = Path(raw).resolve(strict=True)
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError("git_repository_redirected") from exc
    if observed != repo_root:
        raise SecGemmaLeanV32PreflightError("git_repository_redirected")


def _safe_regular_bytes(
    repo_root: Path,
    relative_path: str,
    *,
    maximum_bytes: int,
    error_code: str,
) -> bytes:
    path = repo_root
    details: os.stat_result | None = None
    for part in PurePosixPath(_relative_path(relative_path)).parts:
        path = path / part
        try:
            details = path.lstat()
        except OSError as exc:
            raise SecGemmaLeanV32PreflightError(error_code) from exc
        if stat.S_ISLNK(details.st_mode) or _is_reparse(details):
            raise SecGemmaLeanV32PreflightError(error_code)
    if (
        details is None
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_size < 0
        or details.st_size > maximum_bytes
    ):
        raise SecGemmaLeanV32PreflightError(error_code)
    try:
        if path.resolve(strict=True) != path.absolute():
            raise SecGemmaLeanV32PreflightError(error_code)
        payload = path.read_bytes()
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError(error_code) from exc
    if len(payload) != details.st_size or len(payload) > maximum_bytes:
        raise SecGemmaLeanV32PreflightError(error_code)
    return payload


def _verify_execution_closure(
    repo_root: Path,
    head: str,
    *,
    git_runner: GitRunner = _default_git,
    paths: Sequence[str] = EXECUTED_SOURCE_CLOSURE,
    bind_runner: bool = True,
) -> dict[str, Any]:
    root = _canonical_repo_root(repo_root)
    if bind_runner:
        try:
            actual = Path(__file__).resolve(strict=True)
            expected = (root / Path(RUNNER_SOURCE_PATH)).resolve(strict=True)
        except OSError as exc:
            raise SecGemmaLeanV32PreflightError("execution_root_mismatch") from exc
        if actual != expected:
            raise SecGemmaLeanV32PreflightError("execution_root_mismatch")

    hashes: dict[str, str] = {}
    loaded: list[str] = []
    normalized_paths = tuple(sorted(set(paths)))
    if tuple(paths) != normalized_paths:
        raise SecGemmaLeanV32PreflightError("executed_source_invalid")
    for raw_path in normalized_paths:
        path = _relative_path(raw_path, suffix=".py")
        worktree = _safe_regular_bytes(
            root,
            path,
            maximum_bytes=MAX_SOURCE_BYTES,
            error_code="executed_source_invalid",
        )
        committed = git_runner(root, ("cat-file", "blob", f"{head}:{path}"))
        if committed.returncode != 0 or committed.stdout != worktree:
            raise SecGemmaLeanV32PreflightError("executed_source_invalid")
        hashes[path] = hashlib.sha256(worktree).hexdigest()

        module_name = "agent_benchmark" if path == "agent_benchmark/__init__.py" else path[:-3].replace("/", ".")
        module = sys.modules.get(module_name)
        if module is not None:
            try:
                loaded_path = Path(getattr(module, "__file__", "")).resolve(strict=True)
                expected_path = (root / Path(path)).resolve(strict=True)
            except (OSError, TypeError) as exc:
                raise SecGemmaLeanV32PreflightError("execution_root_mismatch") from exc
            if loaded_path != expected_path:
                raise SecGemmaLeanV32PreflightError("execution_root_mismatch")
            loaded.append(module_name)
    return {
        "raw_worktree_bytes_equal_head_blobs": True,
        "raw_bytes_comparison_catches_crlf_conversion": True,
        "runner_source_path": RUNNER_SOURCE_PATH,
        "source_count": len(hashes),
        "source_sha256s": hashes,
        "source_inventory_sha256": _canonical_sha256(hashes),
        "loaded_source_modules_bound": sorted(loaded),
    }


def _verify_repository(
    repo_root: Path,
    *,
    git_runner: GitRunner = _default_git,
    delta_builder: Callable[..., Mapping[str, Any]] = delta.build_delta_manifest,
    delta_validator: Callable[..., Mapping[str, Any]] = delta.validate_delta_manifest,
    source_verifier: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = _canonical_repo_root(repo_root)
    _assert_git_toplevel(root, git_runner=git_runner)
    branch = _git_text(root, "branch", "--show-current", git_runner=git_runner)
    if branch != BRANCH_NAME:
        raise SecGemmaLeanV32PreflightError("wrong_branch")
    status = git_runner(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    if status.returncode != 0 or status.stdout:
        raise SecGemmaLeanV32PreflightError("working_tree_not_clean")
    origin = _git_text(root, "remote", "get-url", "origin", git_runner=git_runner)
    if origin != EXPECTED_ORIGIN_URL:
        raise SecGemmaLeanV32PreflightError("origin_changed")
    head = _sha1(_git_text(root, "rev-parse", "HEAD", git_runner=git_runner))
    head_tree = _sha1(_git_text(root, "rev-parse", "HEAD^{tree}", git_runner=git_runner))
    upstream_ref = _git_text(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        git_runner=git_runner,
    )
    if upstream_ref != EXPECTED_UPSTREAM:
        raise SecGemmaLeanV32PreflightError("upstream_changed")
    upstream_head = _sha1(
        _git_text(root, "rev-parse", "@{upstream}", git_runner=git_runner)
    )
    if upstream_head != head:
        raise SecGemmaLeanV32PreflightError("head_not_at_cached_upstream")
    if not _git_success(
        root,
        "merge-base",
        "--is-ancestor",
        delta.PREREG_COMMIT,
        head,
        git_runner=git_runner,
    ):
        raise SecGemmaLeanV32PreflightError("preregistration_not_ancestor")
    doc_blob = _sha1(
        _git_text(
            root,
            "rev-parse",
            f"HEAD:{delta.PREREG_DOC_PATH}",
            git_runner=git_runner,
        )
    )
    if doc_blob != delta.PREREG_DOC_BLOB:
        raise SecGemmaLeanV32PreflightError("preregistration_changed")

    try:
        manifest = dict(
            delta_builder(
                root,
                head,
                head_tree,
                allow_spec=delta.DEFAULT_ALLOW_SPEC,
                git_binary=_GIT_EXECUTABLE,
            )
        )
        manifest_bytes = delta.serialize_delta_manifest(manifest)
        validated = dict(
            delta_validator(
                root,
                manifest_bytes,
                expected_implementation_commit=head,
                expected_implementation_tree=head_tree,
                allow_spec=delta.DEFAULT_ALLOW_SPEC,
                git_binary=_GIT_EXECUTABLE,
            )
        )
    except Exception as exc:
        raise SecGemmaLeanV32PreflightError("implementation_delta_invalid") from exc
    if validated != manifest:
        raise SecGemmaLeanV32PreflightError("implementation_delta_invalid")
    if source_verifier is None:
        source_identity = _verify_execution_closure(root, head, git_runner=git_runner)
    else:
        source_identity = dict(source_verifier(root, head))
    return {
        "branch": branch,
        "head_commit": head,
        "head_tree": head_tree,
        "upstream_ref": upstream_ref,
        "cached_upstream_commit": upstream_head,
        "remote_state_basis": "local_cached_origin_tracking_ref",
        "origin_url": origin,
        "working_tree_clean_at_verification": True,
        "preregistration_commit": delta.PREREG_COMMIT,
        "preregistration_tree": delta.PREREG_TREE,
        "preregistration_document_path": delta.PREREG_DOC_PATH,
        "preregistration_document_blob": doc_blob,
        "preregistration_document_sha256": delta.PREREG_DOC_SHA256,
        "implementation_delta_manifest": manifest,
        "implementation_delta_manifest_sha256": manifest["manifest_sha256"],
        "implementation_delta_recomputed_from_committed_blobs": True,
        "executed_source_identity": source_identity,
    }


def _verify_private_contact(
    repo_root: Path,
    *,
    git_runner: GitRunner = _default_git,
    validator: Callable[[str], Any] | None = None,
) -> str:
    root = _canonical_repo_root(repo_root)
    relative = _relative_path(PRIVATE_CONFIG_PATH, suffix=".json")
    if _git_success(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        relative,
        git_runner=git_runner,
    ):
        raise SecGemmaLeanV32PreflightError("private_config_tracked")
    if not _git_success(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        relative,
        git_runner=git_runner,
    ):
        raise SecGemmaLeanV32PreflightError("private_config_not_ignored")
    raw = _safe_regular_bytes(
        root,
        relative,
        maximum_bytes=MAX_PRIVATE_CONFIG_BYTES,
        error_code="private_config_invalid",
    )
    try:
        payload = _strict_json_object(raw, "private_config_invalid")
        if type(payload.get("secrets")) is not dict:
            raise SecGemmaLeanV32PreflightError("private_config_invalid")
        contact = payload["secrets"].get("sec_user_agent")
        if validator is None:
            from agent_benchmark.sec_point_in_time import validate_sec_user_agent

            validator = validate_sec_user_agent
        audit = validator(contact)
        fingerprint = getattr(audit, "sha256", None)
        valid = getattr(audit, "real_contact_validated", None)
    except SecGemmaLeanV32PreflightError:
        raise
    except Exception as exc:
        raise SecGemmaLeanV32PreflightError("private_contact_invalid") from exc
    finally:
        raw = b""
        payload = None
        contact = None
    if valid is not True or type(fingerprint) is not str or _TAGGED_SHA256_RE.fullmatch(fingerprint) is None:
        raise SecGemmaLeanV32PreflightError("private_contact_invalid")
    return fingerprint


def _safe_directory(path: Path, repo_root: Path) -> None:
    try:
        relative = path.absolute().relative_to(repo_root)
    except ValueError as exc:
        raise SecGemmaLeanV32PreflightError("checkpoint_root_invalid") from exc
    current = repo_root
    for part in relative.parts:
        current = current / part
        if not current.exists():
            continue
        try:
            details = current.lstat()
        except OSError as exc:
            raise SecGemmaLeanV32PreflightError("checkpoint_root_invalid") from exc
        if stat.S_ISLNK(details.st_mode) or _is_reparse(details):
            raise SecGemmaLeanV32PreflightError("checkpoint_root_invalid")
        if not stat.S_ISDIR(details.st_mode):
            raise SecGemmaLeanV32PreflightError("checkpoint_root_invalid")


def _prepare_checkpoint_root(
    repo_root: Path,
    *,
    git_runner: GitRunner = _default_git,
) -> Path:
    root = _canonical_repo_root(repo_root)
    relative = _relative_path(CHECKPOINT_ROOT)
    if not _git_success(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        f"{relative}/.ignore-probe",
        git_runner=git_runner,
    ):
        raise SecGemmaLeanV32PreflightError("checkpoint_root_not_ignored")
    checkpoint = root / Path(relative)
    _safe_directory(checkpoint, root)
    try:
        checkpoint.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError("checkpoint_root_unwritable") from exc
    _safe_directory(checkpoint, root)
    return checkpoint


def _lock_byte(handle: Any, *, acquire: bool) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        mode = msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK
        msvcrt.locking(handle.fileno(), mode, 1)
    else:
        import fcntl

        mode = fcntl.LOCK_EX | fcntl.LOCK_NB if acquire else fcntl.LOCK_UN
        fcntl.flock(handle.fileno(), mode)


def _existing_lock_is_available(path: Path) -> bool:
    try:
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or _is_reparse(details)
            or details.st_nlink != 1
            or details.st_size != 1
        ):
            return False
        with path.open("r+b") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (details.st_dev, details.st_ino):
                return False
            _lock_byte(handle, acquire=True)
            _lock_byte(handle, acquire=False)
    except OSError:
        return False
    return True


def _assert_no_active_run(
    checkpoint_root: Path,
    *,
    owned_preflight_lock: Path | None = None,
) -> None:
    try:
        candidates = sorted(
            (
                path
                for path in checkpoint_root.rglob("*")
                if path.name in {ACTIVE_LOCK_NAME, GLOBAL_DISPATCH_LOCK_NAME, RUN_LOCK_NAME}
            ),
            key=lambda path: str(path),
        )
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError("active_run_check_failed") from exc
    for candidate in candidates:
        if (
            owned_preflight_lock is not None
            and candidate == owned_preflight_lock
            and candidate.name == ACTIVE_LOCK_NAME
        ):
            continue
        if candidate.name == ACTIVE_LOCK_NAME or not _existing_lock_is_available(candidate):
            raise SecGemmaLeanV32PreflightError("active_run_present")


class _PreflightLease(AbstractContextManager["_PreflightLease"]):
    def __init__(self, checkpoint_root: Path):
        self._path = checkpoint_root / ACTIVE_LOCK_NAME
        self._handle: Any | None = None
        self._identity: tuple[int, int] | None = None
        self._payload_hash: str | None = None

    def __enter__(self) -> "_PreflightLease":
        payload = _canonical_json(
            {
                "schema_version": "aapl-sec-gemma-lean-evidence-v3-2-preflight-lease-v1",
                "pid": os.getpid(),
                "nonce_sha256": hashlib.sha256(uuid.uuid4().hex.encode("ascii")).hexdigest(),
            }
        )
        try:
            self._handle = self._path.open("xb")
            opened = os.fstat(self._handle.fileno())
            self._identity = (opened.st_dev, opened.st_ino)
            self._payload_hash = hashlib.sha256(payload).hexdigest()
            self._handle.write(payload)
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except FileExistsError as exc:
            raise SecGemmaLeanV32PreflightError("active_run_present") from exc
        except OSError as exc:
            self._best_effort_cleanup()
            raise SecGemmaLeanV32PreflightError("run_lock_unavailable") from exc
        return self

    def _owned(self, *, require_payload: bool = True) -> bool:
        if self._identity is None or (require_payload and self._payload_hash is None):
            return False
        try:
            details = self._path.lstat()
            identity_matches = (
                stat.S_ISREG(details.st_mode)
                and not stat.S_ISLNK(details.st_mode)
                and not _is_reparse(details)
                and details.st_nlink == 1
                and (details.st_dev, details.st_ino) == self._identity
            )
            if not identity_matches or not require_payload:
                return identity_matches
            return hashlib.sha256(self._path.read_bytes()).hexdigest() == self._payload_hash
        except OSError:
            return False

    def _best_effort_cleanup(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        if self._owned(require_payload=False):
            try:
                self._path.unlink()
            except OSError:
                pass

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        owned = self._owned()
        close_failed = False
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                close_failed = True
            self._handle = None
        try:
            if not owned or not self._owned():
                raise OSError("lease identity changed")
            self._path.unlink()
        except OSError:
            raise SecGemmaLeanV32PreflightError("run_lock_cleanup_failed") from None
        if close_failed:
            raise SecGemmaLeanV32PreflightError("run_lock_cleanup_failed")
        return False


def _write_probe(checkpoint_root: Path) -> None:
    path = checkpoint_root / f".write-probe-{uuid.uuid4().hex}"
    expected = b"aapl-sec-gemma-lean-evidence-v3-2-write-probe\n"
    try:
        with path.open("xb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        if path.read_bytes() != expected:
            raise SecGemmaLeanV32PreflightError("checkpoint_probe_mismatch")
    except SecGemmaLeanV32PreflightError:
        raise
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError("checkpoint_root_unwritable") from exc
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise SecGemmaLeanV32PreflightError("checkpoint_probe_cleanup_failed") from exc


def _assert_no_prior_identity_probe(checkpoint_root: Path) -> None:
    for name in (
        PROBE_INTENT_NAME,
        PROBE_COMPLETION_NAME,
        TEST_PROBE_INTENT_NAME,
        TEST_PROBE_COMPLETION_NAME,
        PUBLISH_COMPLETE_NAME,
    ):
        path = checkpoint_root / name
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid") from exc
        raise SecGemmaLeanV32PreflightError("identity_probe_already_attempted")


def _write_durable_marker(
    checkpoint_root: Path,
    name: str,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    if name not in {
        PROBE_INTENT_NAME,
        PROBE_COMPLETION_NAME,
        TEST_PROBE_INTENT_NAME,
        TEST_PROBE_COMPLETION_NAME,
        PUBLISH_COMPLETE_NAME,
    }:
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    marker_body = dict(body)
    marker = {**marker_body, "marker_sha256": _canonical_sha256(marker_body)}
    payload = _canonical_json(marker) + b"\n"
    path = checkpoint_root / name
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(checkpoint_root)
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or _is_reparse(details)
            or details.st_nlink != 1
            or path.read_bytes() != payload
        ):
            raise OSError("durable marker changed")
    except FileExistsError as exc:
        raise SecGemmaLeanV32PreflightError("identity_probe_already_attempted") from exc
    except OSError as exc:
        # Never remove an uncertain marker: its presence is the fail-closed
        # evidence which prevents another identity request after a crash.
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid") from exc
    return {
        "path": f"{CHECKPOINT_ROOT}/{name}",
        "marker_sha256": marker["marker_sha256"],
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _durable_identity_probe_core(
    checkpoint_root: Path,
    repository: Mapping[str, Any],
    *,
    contact_fingerprint: str,
    runtime_verifier: Callable[[], Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record intent before any call and completion only after exact success."""

    _assert_no_prior_identity_probe(checkpoint_root)
    injected = runtime_verifier is not None
    intent_name = TEST_PROBE_INTENT_NAME if injected else PROBE_INTENT_NAME
    completion_name = (
        TEST_PROBE_COMPLETION_NAME if injected else PROBE_COMPLETION_NAME
    )
    if (
        type(contact_fingerprint) is not str
        or _TAGGED_SHA256_RE.fullmatch(contact_fingerprint) is None
    ):
        raise SecGemmaLeanV32PreflightError("private_contact_invalid")
    attempt_nonce_sha256 = hashlib.sha256(uuid.uuid4().hex.encode("ascii")).hexdigest()
    intent_body = {
        "schema_version": (
            "aapl-sec-gemma-lean-evidence-v3-2-test-identity-intent-v1"
            if injected
            else "aapl-sec-gemma-lean-evidence-v3-2-identity-intent-v1"
        ),
        "test_injections_used": injected,
        "implementation_commit": repository.get("head_commit"),
        "implementation_tree": repository.get("head_tree"),
        "implementation_delta_manifest_sha256": repository.get(
            "implementation_delta_manifest_sha256"
        ),
        "source_inventory_sha256": repository.get(
            "executed_source_identity", {}
        ).get("source_inventory_sha256"),
        "private_contact_sha256": contact_fingerprint,
        "attempt_nonce_sha256": attempt_nonce_sha256,
        "loopback_host": LOOPBACK_HOST,
        "loopback_port": LOOPBACK_PORT,
        "request_sequence": [list(item) for item in ALLOWED_IDENTITY_SEQUENCE],
        "request_count": 2,
        "version_requests": 0,
        "chat_or_generation_requests": 0,
        "external_network_requests": 0,
    }
    intent = _write_durable_marker(checkpoint_root, intent_name, intent_body)
    try:
        runtime = dict(
            _verify_runtime_identity(trusted=True)
            if runtime_verifier is None
            else runtime_verifier()
        )
    except SecGemmaLeanV32PreflightError:
        raise
    except Exception as exc:
        raise SecGemmaLeanV32PreflightError("runtime_identity_invalid") from exc
    probe = runtime.get("identity_probe")
    if (
        type(probe) is not dict
        or probe.get("request_sequence") != [list(item) for item in ALLOWED_IDENTITY_SEQUENCE]
        or probe.get("observed_request_count") != 2
        or probe.get("version_requests") != 0
        or probe.get("chat_or_generation_requests") != 0
        or probe.get("test_injections_used") is not injected
    ):
        raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
    completion_body = {
        "schema_version": (
            "aapl-sec-gemma-lean-evidence-v3-2-test-identity-complete-v1"
            if injected
            else "aapl-sec-gemma-lean-evidence-v3-2-identity-complete-v1"
        ),
        "test_injections_used": injected,
        "attempt_nonce_sha256": attempt_nonce_sha256,
        "intent_marker_sha256": intent["marker_sha256"],
        "intent_file_sha256": intent["file_sha256"],
        "request_sequence": [list(item) for item in ALLOWED_IDENTITY_SEQUENCE],
        "request_count": 2,
        "tags_response_sha256": runtime.get("tags_identity", {}).get(
            "tags_response_sha256"
        ),
        "show_response_raw_sha256": runtime.get(
            "ollama_show_raw_sha256_diagnostic"
        ),
        "show_response_semantic_sha256": runtime.get(
            "ollama_show_semantic_sha256"
        ),
        "runtime_receipt_sha256": runtime.get("runtime_receipt_sha256"),
        "runtime_fingerprint_sha256": runtime.get("runtime_fingerprint_sha256"),
        "model_manifest_sha256": runtime.get("model_manifest_sha256"),
        "response_bodies_retained": False,
        "identity_verified": True,
    }
    completion = _write_durable_marker(
        checkpoint_root,
        completion_name,
        completion_body,
    )
    return runtime, {
        "intent": intent,
        "completion": completion,
        "request_sequence": [list(item) for item in ALLOWED_IDENTITY_SEQUENCE],
        "markers_retained_after_success": True,
        "retry_after_any_intent_is_forbidden": True,
        "response_bodies_retained": False,
    }


def _durable_identity_probe(
    checkpoint_root: Path,
    repository: Mapping[str, Any],
    *,
    contact_fingerprint: str,
    runtime_verifier: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Test-only injected durable probe helper; never used by production."""

    return _durable_identity_probe_core(
        checkpoint_root,
        repository,
        contact_fingerprint=contact_fingerprint,
        runtime_verifier=runtime_verifier,
    )


def _durable_production_identity_probe(
    checkpoint_root: Path,
    repository: Mapping[str, Any],
    contact_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trusted path with the fixed identity routine and no injected callback."""

    return _durable_identity_probe_core(
        checkpoint_root,
        repository,
        contact_fingerprint=contact_fingerprint,
        runtime_verifier=None,
    )


class _LoopbackJSONClient:
    """One-call HTTP delegate pinned to literal loopback; no proxy or redirect."""

    def request(self, method: str, path: str, *, body: bytes) -> bytes:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
            "Content-Type": "application/json",
            "User-Agent": "aapl-sec-gemma-lean-v3.2-preflight/1",
        }
        connection = http.client.HTTPConnection(
            LOOPBACK_HOST,
            LOOPBACK_PORT,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response: Any | None = None
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            encoding = response.getheader("Content-Encoding", "").strip().lower()
            if encoding not in {"", "identity"}:
                raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            lengths = [value.strip() for key, value in response.getheaders() if key.lower() == "content-length"]
            transfers = [value.strip().lower() for key, value in response.getheaders() if key.lower() == "transfer-encoding"]
            if len(lengths) > 1 or len(transfers) > 1:
                raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            if lengths and transfers:
                raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            if transfers and transfers != ["chunked"]:
                raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            if lengths:
                try:
                    declared = int(lengths[0], 10)
                except ValueError as exc:
                    raise SecGemmaLeanV32PreflightError("runtime_identity_invalid") from exc
                if declared < 1 or declared > MAX_IDENTITY_RESPONSE_BYTES:
                    raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            payload = response.read(MAX_IDENTITY_RESPONSE_BYTES + 1)
            if not payload or len(payload) > MAX_IDENTITY_RESPONSE_BYTES:
                raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            if lengths and len(payload) != declared:
                raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
            return bytes(payload)
        except SecGemmaLeanV32PreflightError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise SecGemmaLeanV32PreflightError("runtime_identity_invalid") from exc
        finally:
            connection.close()


class _RestrictedIdentityTransport:
    """Permit and record only the exact two-call identity sequence."""

    def __init__(self, delegate: Any):
        if not callable(getattr(delegate, "request", None)):
            raise SecGemmaLeanV32PreflightError("runtime_transport_invalid")
        self._delegate = delegate
        self._calls: list[tuple[str, str]] = []

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._calls)

    def request(self, method: str, path: str, *, body: bytes) -> bytes:
        position = len(self._calls)
        if position >= len(ALLOWED_IDENTITY_SEQUENCE) or (method, path) != ALLOWED_IDENTITY_SEQUENCE[position]:
            raise SecGemmaLeanV32PreflightError("runtime_probe_request_forbidden")
        expected_body = b"" if position == 0 else SHOW_REQUEST_BYTES
        if type(body) is not bytes or body != expected_body:
            raise SecGemmaLeanV32PreflightError("runtime_probe_request_forbidden")
        self._calls.append((method, path))
        result = self._delegate.request(method, path, body=body)
        if type(result) is not bytes:
            raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
        return result


def _verify_tags(tags_bytes: bytes) -> dict[str, Any]:
    payload = _strict_json_object(tags_bytes, "runtime_tags_invalid")
    if set(payload) != {"models"} or type(payload["models"]) is not list:
        raise SecGemmaLeanV32PreflightError("runtime_tags_invalid")
    matching: list[dict[str, Any]] = []
    for candidate in payload["models"]:
        if type(candidate) is not dict:
            raise SecGemmaLeanV32PreflightError("runtime_tags_invalid")
        names = [candidate.get("name"), candidate.get("model")]
        if any(value == MODEL_NAME for value in names):
            matching.append(candidate)
    if len(matching) != 1:
        raise SecGemmaLeanV32PreflightError("runtime_tags_invalid")
    selected = matching[0]
    for key in ("name", "model"):
        if key in selected and selected[key] != MODEL_NAME:
            raise SecGemmaLeanV32PreflightError("runtime_tags_invalid")
    if selected.get("digest") != MODEL_MANIFEST_SHA256:
        raise SecGemmaLeanV32PreflightError("runtime_tags_invalid")
    return {
        "model_name": MODEL_NAME,
        "unique_model_match": True,
        "manifest_digest": MODEL_MANIFEST_SHA256,
        "tags_response_sha256": hashlib.sha256(tags_bytes).hexdigest(),
    }


def _verify_runtime_identity(
    *,
    delegate: Any | None = None,
    bundle_verifier: Callable[..., Mapping[str, Any]] | None = None,
    trusted: bool = False,
) -> dict[str, Any]:
    if trusted and (delegate is not None or bundle_verifier is not None):
        raise SecGemmaLeanV32PreflightError("runtime_transport_invalid")
    transport = _RestrictedIdentityTransport(
        _LoopbackJSONClient() if delegate is None else delegate
    )
    try:
        tags_bytes = transport.request("GET", TAGS_PATH, body=b"")
        show_bytes = transport.request("POST", SHOW_PATH, body=SHOW_REQUEST_BYTES)
        tags_receipt = _verify_tags(tags_bytes)
        if bundle_verifier is None:
            from agent_benchmark.sec_gemma_online_risk_overlay_runtime import (
                verify_installed_pinned_runtime,
            )

            bundle_verifier = verify_installed_pinned_runtime
        runtime = dict(
            bundle_verifier(
                version_response_bytes=OLLAMA_VERSION_CANONICAL_BYTES,
                show_response_bytes=show_bytes,
            )
        )
    except SecGemmaLeanV32PreflightError:
        raise
    except Exception as exc:
        raise SecGemmaLeanV32PreflightError("runtime_identity_invalid") from exc
    if transport.calls != ALLOWED_IDENTITY_SEQUENCE:
        raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
    if (
        hashlib.sha256(OLLAMA_VERSION_CANONICAL_BYTES).hexdigest() != OLLAMA_VERSION_RESPONSE_SHA256
        or runtime.get("model_name") != MODEL_NAME
        or runtime.get("manifest_sha256") != MODEL_MANIFEST_SHA256
        or runtime.get("config_sha256") != MODEL_CONFIG_DIGEST
        or tuple(runtime.get("ordered_layer_content_sha256s", ())) != MODEL_LAYER_DIGESTS
        or runtime.get("version_response_sha256") != OLLAMA_VERSION_RESPONSE_SHA256
        or runtime.get("show_response_semantic_sha256") != OLLAMA_SHOW_SEMANTIC_SHA256
        or runtime.get("model_info_sha256") != MODEL_INFO_SHA256
        or runtime.get("runtime_fingerprint_sha256") != RUNTIME_FINGERPRINT_SHA256
        or type(runtime.get("runtime_receipt_sha256")) is not str
        or _SHA256_RE.fullmatch(runtime["runtime_receipt_sha256"]) is None
    ):
        raise SecGemmaLeanV32PreflightError("runtime_identity_invalid")
    return {
        "model_name": MODEL_NAME,
        "runtime_fingerprint_sha256": RUNTIME_FINGERPRINT_SHA256,
        "runtime_receipt_sha256": runtime["runtime_receipt_sha256"],
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "model_config_sha256": MODEL_CONFIG_DIGEST,
        "ordered_layer_content_sha256s": list(MODEL_LAYER_DIGESTS),
        "ollama_version": OLLAMA_VERSION,
        "ollama_version_response_sha256": OLLAMA_VERSION_RESPONSE_SHA256,
        "ollama_show_semantic_sha256": OLLAMA_SHOW_SEMANTIC_SHA256,
        "ollama_show_raw_sha256_diagnostic": runtime.get("show_response_raw_sha256"),
        "model_info_sha256": MODEL_INFO_SHA256,
        "tags_identity": tags_receipt,
        "version_identity_basis": {
            "live_api_version_call_made": False,
            "canonical_response_constructed_locally": True,
            "canonical_response_ascii": OLLAMA_VERSION_CANONICAL_BYTES.decode("ascii"),
            "basis": "inherited_preregistered_ollama_version_pin",
        },
        "identity_probe": {
            "loopback_host": LOOPBACK_HOST,
            "loopback_port": LOOPBACK_PORT,
            "restricted_sequence_enforced": True,
            "request_sequence": [list(item) for item in transport.calls],
            "observed_request_count": len(transport.calls),
            "tags_requests": 1,
            "show_requests": 1,
            "version_requests": 0,
            "chat_or_generation_requests": 0,
            "test_injections_used": not trusted,
        },
    }


def _effect_counts() -> dict[str, int]:
    return {
        "official_sec_requests": 0,
        "yahoo_requests": 0,
        "market_data_requests": 0,
        "model_generation_calls": 0,
        "performance_calculations": 0,
        "performance_results_opened": 0,
        "trade_actions": 0,
        "broker_actions": 0,
        "real_money_actions": 0,
        "paid_api_calls": 0,
        "external_network_requests": 0,
        "loopback_identity_requests": 2,
        "loopback_tags_requests": 1,
        "loopback_show_requests": 1,
        "loopback_version_requests": 0,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_repository(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    identity = dict(result.get("executed_source_identity", {}))
    identity.pop("loaded_source_modules_bound", None)
    result["executed_source_identity"] = identity
    return result


def _build_report(
    *,
    repository_before: Mapping[str, Any],
    repository_after: Mapping[str, Any],
    contact_fingerprint: str,
    runtime: Mapping[str, Any],
    created_at: str,
    elapsed_seconds: float,
    trusted: bool,
    artifact_will_dirty: bool,
    probe_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    before_hash = _canonical_sha256(_stable_repository(repository_before))
    after_hash = _canonical_sha256(_stable_repository(repository_after))
    if before_hash != after_hash:
        raise SecGemmaLeanV32PreflightError("repository_changed_during_preflight")
    try:
        parsed_time = datetime.fromisoformat(created_at[:-1] + "+00:00")
    except (TypeError, ValueError):
        parsed_time = None
    if (
        type(created_at) is not str
        or _UTC_RE.fullmatch(created_at) is None
        or parsed_time is None
        or parsed_time.tzinfo != timezone.utc
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0.0
    ):
        raise SecGemmaLeanV32PreflightError("clock_invalid")
    repository = dict(repository_after)
    repository.update(
        {
            "reverified_after_identity_probe": True,
            "pre_identity_stable_repository_sha256": before_hash,
            "post_identity_stable_repository_sha256": after_hash,
        }
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if trusted else "test_only_untrusted",
        "created_at_utc": created_at,
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "repository": repository,
        "repository_observation_timing": {
            "cleanliness_observed_before_artifact_write": True,
            "post_artifact_cleanliness_claimed": False,
            "artifact_write_will_dirty_worktree_until_commit": artifact_will_dirty,
        },
        "private_sec_contact": {
            "sha256": contact_fingerprint,
            "plausible_format_validated": True,
            "reachability_not_proven": True,
            "readable_value_published": False,
        },
        "local_runtime": dict(runtime),
        "checkpoint": {
            "root": CHECKPOINT_ROOT,
            "ignored_local_state": True,
            "exclusive_preflight_lease_held_during_checks": True,
            "no_active_run_observed_before_and_after": True,
            "write_flush_read_delete_probe_passed": True,
            "probe_file_retained": False,
            "lease_released_after_return": True,
            "acquisition_must_reacquire_and_reverify": True,
            "durable_identity_probe_evidence": (
                dict(probe_evidence) if trusted and probe_evidence is not None else None
            ),
        },
        "effect_counts": _effect_counts() if trusted else {key: None for key in _effect_counts()},
        "effect_counts_basis": {
            "trusted": trusted,
            "exact_two_loopback_calls_recorded_by_restricted_transport": trusted,
            "all_other_counts_bound_by_fixed_zero_effect_code_path": trusted,
        },
        "trust": {
            "production_defaults_only": trusted,
            "test_injections_used": not trusted,
            "eligible_for_public_seal": trusted,
        },
        "execution_authorized": False,
        "next_step": "development_source_acquisition",
    }
    return {**body, "preflight_sha256": _canonical_sha256(body)}


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
    """Run injected offline checks; the result is permanently unsealable."""

    root = _canonical_repo_root(repo_root)
    try:
        started = float(clock())
        checkpoint = checkpoint_preparer(root)
        with _PreflightLease(checkpoint):
            before = dict(repository_verifier(root))
            fingerprint = contact_verifier(root)
            runtime = dict(runtime_verifier())
            after = dict(repository_verifier(root))
            _write_probe(checkpoint)
            finished = float(clock())
            created = now()
            return _build_report(
                repository_before=before,
                repository_after=after,
                contact_fingerprint=fingerprint,
                runtime=runtime,
                created_at=created,
                elapsed_seconds=finished - started,
                trusted=False,
                artifact_will_dirty=False,
                probe_evidence=None,
            )
    except SecGemmaLeanV32PreflightError:
        raise
    except Exception as exc:
        raise SecGemmaLeanV32PreflightError("untrusted_preflight_failed") from exc


def _validate_sealable_report(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaLeanV32PreflightError("untrusted_report_not_sealable")
    report = value
    body = {key: child for key, child in report.items() if key != "preflight_sha256"}
    repository = report.get("repository")
    runtime = report.get("local_runtime")
    contact = report.get("private_sec_contact")
    checkpoint = report.get("checkpoint")
    probe = runtime.get("identity_probe", {}) if type(runtime) is dict else {}
    tags_identity = runtime.get("tags_identity", {}) if type(runtime) is dict else {}
    if type(tags_identity) is not dict:
        tags_identity = {}
    durable = (
        checkpoint.get("durable_identity_probe_evidence", {})
        if type(checkpoint) is dict
        else {}
    )
    source = repository.get("executed_source_identity", {}) if type(repository) is dict else {}
    implementation_manifest = (
        repository.get("implementation_delta_manifest", {})
        if type(repository) is dict
        else {}
    )
    loaded = source.get("loaded_source_modules_bound", []) if type(source) is dict else []
    source_hashes = source.get("source_sha256s", {}) if type(source) is dict else {}
    required_loaded = {
        "agent_benchmark",
        "agent_benchmark.sec_gemma_lean_v32_delta",
        "agent_benchmark.sec_point_in_time",
        "agent_benchmark.sec_gemma_online_risk_overlay_runtime",
        "agent_benchmark.sec_gemma_online_risk_overlay_contract",
        "agent_benchmark.sec_filing_gemma_extractor_prompt",
        "agent_benchmark.sec_filing_gemma_extractor_schema",
    }
    expected_top_level = {
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
    expected_runtime_keys = {
        "model_name",
        "runtime_fingerprint_sha256",
        "runtime_receipt_sha256",
        "model_manifest_sha256",
        "model_config_sha256",
        "ordered_layer_content_sha256s",
        "ollama_version",
        "ollama_version_response_sha256",
        "ollama_show_semantic_sha256",
        "ollama_show_raw_sha256_diagnostic",
        "model_info_sha256",
        "tags_identity",
        "version_identity_basis",
        "identity_probe",
    }
    expected_checkpoint_keys = {
        "root",
        "ignored_local_state",
        "exclusive_preflight_lease_held_during_checks",
        "no_active_run_observed_before_and_after",
        "write_flush_read_delete_probe_passed",
        "probe_file_retained",
        "lease_released_after_return",
        "acquisition_must_reacquire_and_reverify",
        "durable_identity_probe_evidence",
    }
    expected_repository_keys = {
        "branch",
        "head_commit",
        "head_tree",
        "upstream_ref",
        "cached_upstream_commit",
        "remote_state_basis",
        "origin_url",
        "working_tree_clean_at_verification",
        "preregistration_commit",
        "preregistration_tree",
        "preregistration_document_path",
        "preregistration_document_blob",
        "preregistration_document_sha256",
        "implementation_delta_manifest",
        "implementation_delta_manifest_sha256",
        "implementation_delta_recomputed_from_committed_blobs",
        "executed_source_identity",
        "reverified_after_identity_probe",
        "pre_identity_stable_repository_sha256",
        "post_identity_stable_repository_sha256",
    }
    expected_source_keys = {
        "raw_worktree_bytes_equal_head_blobs",
        "raw_bytes_comparison_catches_crlf_conversion",
        "runner_source_path",
        "source_count",
        "source_sha256s",
        "source_inventory_sha256",
        "loaded_source_modules_bound",
    }
    try:
        elapsed = float.fromhex(report.get("elapsed_seconds_hex", ""))
    except (TypeError, ValueError):
        elapsed = -1.0
    valid = (
        set(report) == expected_top_level
        and report.get("schema_version") == SCHEMA_VERSION
        and report.get("status") == "passed"
        and report.get("preflight_sha256") == _canonical_sha256(body)
        and type(report.get("created_at_utc")) is str
        and _UTC_RE.fullmatch(report["created_at_utc"]) is not None
        and math.isfinite(elapsed)
        and elapsed >= 0.0
        and report.get("effect_counts") == _effect_counts()
        and report.get("trust")
        == {
            "production_defaults_only": True,
            "test_injections_used": False,
            "eligible_for_public_seal": True,
        }
        and report.get("effect_counts_basis")
        == {
            "trusted": True,
            "exact_two_loopback_calls_recorded_by_restricted_transport": True,
            "all_other_counts_bound_by_fixed_zero_effect_code_path": True,
        }
        and report.get("execution_authorized") is False
        and report.get("next_step") == "development_source_acquisition"
        and report.get("repository_observation_timing")
        == {
            "cleanliness_observed_before_artifact_write": True,
            "post_artifact_cleanliness_claimed": False,
            "artifact_write_will_dirty_worktree_until_commit": True,
        }
        and type(repository) is dict
        and set(repository) == expected_repository_keys
        and repository.get("branch") == BRANCH_NAME
        and repository.get("upstream_ref") == EXPECTED_UPSTREAM
        and repository.get("head_commit") == repository.get("cached_upstream_commit")
        and repository.get("working_tree_clean_at_verification") is True
        and repository.get("preregistration_commit") == delta.PREREG_COMMIT
        and repository.get("preregistration_document_blob") == delta.PREREG_DOC_BLOB
        and repository.get("preregistration_tree") == delta.PREREG_TREE
        and repository.get("preregistration_document_path") == delta.PREREG_DOC_PATH
        and repository.get("preregistration_document_sha256") == delta.PREREG_DOC_SHA256
        and type(implementation_manifest) is dict
        and implementation_manifest.get("implementation")
        == {
            "commit": repository.get("head_commit"),
            "tree": repository.get("head_tree"),
        }
        and implementation_manifest.get("manifest_sha256")
        == repository.get("implementation_delta_manifest_sha256")
        and repository.get("implementation_delta_recomputed_from_committed_blobs") is True
        and repository.get("reverified_after_identity_probe") is True
        and repository.get("pre_identity_stable_repository_sha256")
        == repository.get("post_identity_stable_repository_sha256")
        and type(source) is dict
        and set(source) == expected_source_keys
        and source.get("raw_worktree_bytes_equal_head_blobs") is True
        and source.get("raw_bytes_comparison_catches_crlf_conversion") is True
        and source.get("runner_source_path") == RUNNER_SOURCE_PATH
        and source.get("source_count") == len(EXECUTED_SOURCE_CLOSURE)
        and type(source_hashes) is dict
        and set(source_hashes) == set(EXECUTED_SOURCE_CLOSURE)
        and all(
            type(item) is str and _SHA256_RE.fullmatch(item) is not None
            for item in source_hashes.values()
        )
        and source.get("source_inventory_sha256") == _canonical_sha256(source_hashes)
        and type(loaded) is list
        and required_loaded.issubset(set(loaded))
        and type(runtime) is dict
        and set(runtime) == expected_runtime_keys
        and runtime.get("model_name") == MODEL_NAME
        and runtime.get("runtime_fingerprint_sha256") == RUNTIME_FINGERPRINT_SHA256
        and runtime.get("model_manifest_sha256") == MODEL_MANIFEST_SHA256
        and runtime.get("model_config_sha256") == MODEL_CONFIG_DIGEST
        and runtime.get("ordered_layer_content_sha256s") == list(MODEL_LAYER_DIGESTS)
        and runtime.get("ollama_version") == OLLAMA_VERSION
        and runtime.get("ollama_version_response_sha256")
        == OLLAMA_VERSION_RESPONSE_SHA256
        and runtime.get("ollama_show_semantic_sha256")
        == OLLAMA_SHOW_SEMANTIC_SHA256
        and runtime.get("model_info_sha256") == MODEL_INFO_SHA256
        and type(runtime.get("runtime_receipt_sha256")) is str
        and _SHA256_RE.fullmatch(runtime["runtime_receipt_sha256"]) is not None
        and type(runtime.get("ollama_show_raw_sha256_diagnostic")) is str
        and _SHA256_RE.fullmatch(runtime["ollama_show_raw_sha256_diagnostic"])
        is not None
        and runtime.get("tags_identity")
        == {
            "model_name": MODEL_NAME,
            "unique_model_match": True,
            "manifest_digest": MODEL_MANIFEST_SHA256,
            "tags_response_sha256": tags_identity.get("tags_response_sha256"),
        }
        and _SHA256_RE.fullmatch(
            tags_identity.get("tags_response_sha256", "")
        )
        is not None
        and set(probe)
        == {
            "loopback_host",
            "loopback_port",
            "restricted_sequence_enforced",
            "request_sequence",
            "observed_request_count",
            "tags_requests",
            "show_requests",
            "version_requests",
            "chat_or_generation_requests",
            "test_injections_used",
        }
        and probe.get("loopback_host") == LOOPBACK_HOST
        and probe.get("loopback_port") == LOOPBACK_PORT
        and probe.get("restricted_sequence_enforced") is True
        and probe.get("request_sequence") == [list(item) for item in ALLOWED_IDENTITY_SEQUENCE]
        and probe.get("observed_request_count") == 2
        and probe.get("tags_requests") == 1
        and probe.get("show_requests") == 1
        and probe.get("version_requests") == 0
        and probe.get("chat_or_generation_requests") == 0
        and probe.get("test_injections_used") is False
        and runtime.get("version_identity_basis")
        == {
            "live_api_version_call_made": False,
            "canonical_response_constructed_locally": True,
            "canonical_response_ascii": OLLAMA_VERSION_CANONICAL_BYTES.decode("ascii"),
            "basis": "inherited_preregistered_ollama_version_pin",
        }
        and type(checkpoint) is dict
        and set(checkpoint) == expected_checkpoint_keys
        and checkpoint.get("root") == CHECKPOINT_ROOT
        and checkpoint.get("ignored_local_state") is True
        and checkpoint.get("exclusive_preflight_lease_held_during_checks") is True
        and checkpoint.get("no_active_run_observed_before_and_after") is True
        and checkpoint.get("write_flush_read_delete_probe_passed") is True
        and checkpoint.get("probe_file_retained") is False
        and checkpoint.get("lease_released_after_return") is True
        and checkpoint.get("acquisition_must_reacquire_and_reverify") is True
        and type(durable) is dict
        and set(durable)
        == {
            "intent",
            "completion",
            "request_sequence",
            "markers_retained_after_success",
            "retry_after_any_intent_is_forbidden",
            "response_bodies_retained",
        }
        and durable.get("request_sequence")
        == [list(item) for item in ALLOWED_IDENTITY_SEQUENCE]
        and durable.get("markers_retained_after_success") is True
        and durable.get("retry_after_any_intent_is_forbidden") is True
        and durable.get("response_bodies_retained") is False
        and type(durable.get("intent")) is dict
        and durable["intent"].get("path")
        == f"{CHECKPOINT_ROOT}/{PROBE_INTENT_NAME}"
        and _SHA256_RE.fullmatch(durable["intent"].get("marker_sha256", "")) is not None
        and _SHA256_RE.fullmatch(durable["intent"].get("file_sha256", "")) is not None
        and type(durable.get("completion")) is dict
        and durable["completion"].get("path")
        == f"{CHECKPOINT_ROOT}/{PROBE_COMPLETION_NAME}"
        and _SHA256_RE.fullmatch(durable["completion"].get("marker_sha256", "")) is not None
        and _SHA256_RE.fullmatch(durable["completion"].get("file_sha256", "")) is not None
        and type(contact) is dict
        and set(contact)
        == {
            "sha256",
            "plausible_format_validated",
            "reachability_not_proven",
            "readable_value_published",
        }
        and _TAGGED_SHA256_RE.fullmatch(contact.get("sha256", "")) is not None
        and contact.get("plausible_format_validated") is True
        and contact.get("reachability_not_proven") is True
        and contact.get("readable_value_published") is False
    )
    if not valid:
        raise SecGemmaLeanV32PreflightError("untrusted_report_not_sealable")
    return report


def _read_verified_probe_marker(
    repo_root: Path,
    name: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if name not in {PROBE_INTENT_NAME, PROBE_COMPLETION_NAME} or type(evidence) is not dict:
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    expected_path = f"{CHECKPOINT_ROOT}/{name}"
    if (
        set(evidence) != {"path", "marker_sha256", "file_sha256"}
        or evidence.get("path") != expected_path
        or _SHA256_RE.fullmatch(evidence.get("marker_sha256", "")) is None
        or _SHA256_RE.fullmatch(evidence.get("file_sha256", "")) is None
    ):
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    payload = _safe_regular_bytes(
        repo_root,
        expected_path,
        maximum_bytes=64 * 1024,
        error_code="identity_probe_state_invalid",
    )
    if hashlib.sha256(payload).hexdigest() != evidence["file_sha256"]:
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    marker = _strict_json_object(payload, "identity_probe_state_invalid")
    if payload != _canonical_json(marker) + b"\n":
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    supplied = marker.get("marker_sha256")
    body = {key: value for key, value in marker.items() if key != "marker_sha256"}
    if supplied != evidence["marker_sha256"] or supplied != _canonical_sha256(body):
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    return marker


def _validate_durable_probe_state(
    repo_root: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    runtime = report.get("local_runtime")
    repository = report.get("repository")
    if type(checkpoint) is not dict or type(runtime) is not dict or type(repository) is not dict:
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    evidence = checkpoint.get("durable_identity_probe_evidence")
    if type(evidence) is not dict:
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    intent = _read_verified_probe_marker(repo_root, PROBE_INTENT_NAME, evidence.get("intent", {}))
    completion = _read_verified_probe_marker(
        repo_root, PROBE_COMPLETION_NAME, evidence.get("completion", {})
    )
    expected_intent_keys = {
        "schema_version",
        "test_injections_used",
        "implementation_commit",
        "implementation_tree",
        "implementation_delta_manifest_sha256",
        "source_inventory_sha256",
        "private_contact_sha256",
        "attempt_nonce_sha256",
        "loopback_host",
        "loopback_port",
        "request_sequence",
        "request_count",
        "version_requests",
        "chat_or_generation_requests",
        "external_network_requests",
        "marker_sha256",
    }
    expected_completion_keys = {
        "schema_version",
        "test_injections_used",
        "attempt_nonce_sha256",
        "intent_marker_sha256",
        "intent_file_sha256",
        "request_sequence",
        "request_count",
        "tags_response_sha256",
        "show_response_raw_sha256",
        "show_response_semantic_sha256",
        "runtime_receipt_sha256",
        "runtime_fingerprint_sha256",
        "model_manifest_sha256",
        "response_bodies_retained",
        "identity_verified",
        "marker_sha256",
    }
    valid = (
        set(intent) == expected_intent_keys
        and intent.get("schema_version")
        == "aapl-sec-gemma-lean-evidence-v3-2-identity-intent-v1"
        and intent.get("test_injections_used") is False
        and intent.get("implementation_commit") == repository.get("head_commit")
        and intent.get("implementation_tree") == repository.get("head_tree")
        and intent.get("implementation_delta_manifest_sha256")
        == repository.get("implementation_delta_manifest_sha256")
        and intent.get("source_inventory_sha256")
        == repository.get("executed_source_identity", {}).get(
            "source_inventory_sha256"
        )
        and intent.get("private_contact_sha256")
        == report.get("private_sec_contact", {}).get("sha256")
        and _SHA256_RE.fullmatch(intent.get("attempt_nonce_sha256", "")) is not None
        and intent.get("loopback_host") == LOOPBACK_HOST
        and intent.get("loopback_port") == LOOPBACK_PORT
        and intent.get("request_sequence") == [list(item) for item in ALLOWED_IDENTITY_SEQUENCE]
        and intent.get("request_count") == 2
        and intent.get("version_requests") == 0
        and intent.get("chat_or_generation_requests") == 0
        and intent.get("external_network_requests") == 0
        and set(completion) == expected_completion_keys
        and completion.get("schema_version")
        == "aapl-sec-gemma-lean-evidence-v3-2-identity-complete-v1"
        and completion.get("test_injections_used") is False
        and completion.get("attempt_nonce_sha256") == intent.get("attempt_nonce_sha256")
        and completion.get("intent_marker_sha256")
        == evidence["intent"]["marker_sha256"]
        and completion.get("intent_file_sha256") == evidence["intent"]["file_sha256"]
        and completion.get("request_sequence")
        == [list(item) for item in ALLOWED_IDENTITY_SEQUENCE]
        and completion.get("request_count") == 2
        and completion.get("tags_response_sha256")
        == runtime.get("tags_identity", {}).get("tags_response_sha256")
        and completion.get("show_response_raw_sha256")
        == runtime.get("ollama_show_raw_sha256_diagnostic")
        and completion.get("show_response_semantic_sha256")
        == runtime.get("ollama_show_semantic_sha256")
        and completion.get("runtime_receipt_sha256") == runtime.get("runtime_receipt_sha256")
        and completion.get("runtime_fingerprint_sha256")
        == runtime.get("runtime_fingerprint_sha256")
        and completion.get("model_manifest_sha256") == MODEL_MANIFEST_SHA256
        and completion.get("response_bodies_retained") is False
        and completion.get("identity_verified") is True
    )
    if not valid:
        raise SecGemmaLeanV32PreflightError("identity_probe_state_invalid")
    return {
        "intent_marker_sha256": evidence["intent"]["marker_sha256"],
        "completion_marker_sha256": evidence["completion"]["marker_sha256"],
        "exact_request_sequence_verified": True,
        "response_bodies_absent": True,
    }


def _validate_committed_preflight_for_acquisition(
    repo_root: Path,
    *,
    git_runner: GitRunner = _default_git,
    delta_builder: Callable[..., Mapping[str, Any]] = delta.build_delta_manifest,
    delta_validator: Callable[..., Mapping[str, Any]] = delta.validate_delta_manifest,
    contact_verifier: Callable[[Path], str] = _verify_private_contact,
    source_verifier: Callable[[Path, str], Mapping[str, Any]] | None = None,
    marker_verifier: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] = (
        _validate_durable_probe_state
    ),
    publish_verifier: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = _canonical_repo_root(repo_root)
    if publish_verifier is None:
        publish_verifier = _validate_publish_complete

    def current_snapshot() -> dict[str, str]:
        _assert_git_toplevel(root, git_runner=git_runner)
        branch = _git_text(root, "branch", "--show-current", git_runner=git_runner)
        status = git_runner(root, ("status", "--porcelain=v1", "--untracked-files=all"))
        origin = _git_text(root, "remote", "get-url", "origin", git_runner=git_runner)
        head = _sha1(_git_text(root, "rev-parse", "HEAD", git_runner=git_runner))
        tree = _sha1(_git_text(root, "rev-parse", "HEAD^{tree}", git_runner=git_runner))
        upstream_ref = _git_text(
            root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            git_runner=git_runner,
        )
        upstream_head = _sha1(
            _git_text(root, "rev-parse", "@{upstream}", git_runner=git_runner)
        )
        if (
            branch != BRANCH_NAME
            or status.returncode != 0
            or status.stdout
            or origin != EXPECTED_ORIGIN_URL
            or upstream_ref != EXPECTED_UPSTREAM
            or head != upstream_head
        ):
            raise SecGemmaLeanV32PreflightError("acquisition_repository_invalid")
        return {"head": head, "tree": tree}

    initial_snapshot = current_snapshot()
    evidence_head = initial_snapshot["head"]
    evidence_tree = initial_snapshot["tree"]

    relative = _relative_path(PREFLIGHT_ARTIFACT_PATH, suffix=".json")
    artifact_bytes = _safe_regular_bytes(
        root,
        relative,
        maximum_bytes=16 * 1024 * 1024,
        error_code="committed_preflight_invalid",
    )
    committed = git_runner(root, ("cat-file", "blob", f"{evidence_head}:{relative}"))
    if committed.returncode != 0 or committed.stdout != artifact_bytes:
        raise SecGemmaLeanV32PreflightError("committed_preflight_invalid")
    report = _validate_sealable_report(
        _strict_json_object(artifact_bytes, "committed_preflight_invalid")
    )
    if artifact_bytes != _serialize_preflight_report(report):
        raise SecGemmaLeanV32PreflightError("committed_preflight_invalid")
    repository = report["repository"]
    implementation_commit = _sha1(repository["head_commit"])
    implementation_tree = _sha1(repository["head_tree"])
    if not _git_success(
        root,
        "merge-base",
        "--is-ancestor",
        implementation_commit,
        evidence_head,
        git_runner=git_runner,
    ):
        raise SecGemmaLeanV32PreflightError("evidence_commit_invalid")
    if not _git_success(
        root,
        "merge-base",
        "--is-ancestor",
        delta.PREREG_COMMIT,
        implementation_commit,
        git_runner=git_runner,
    ):
        raise SecGemmaLeanV32PreflightError("preregistration_not_ancestor")
    parents = _git_text(
        root, "rev-list", "--parents", "-n", "1", evidence_head, git_runner=git_runner
    ).split()
    if parents != [evidence_head, implementation_commit]:
        raise SecGemmaLeanV32PreflightError("evidence_commit_invalid")
    evidence_diff = git_runner(
        root,
        (
            "diff-tree",
            "--name-status",
            "-r",
            "-z",
            "--no-renames",
            implementation_commit,
            evidence_head,
        ),
    )
    expected_diff = b"A\0" + relative.encode("utf-8") + b"\0"
    if evidence_diff.returncode != 0 or evidence_diff.stdout != expected_diff:
        raise SecGemmaLeanV32PreflightError("evidence_commit_not_artifact_only")
    try:
        recomputed = dict(
            delta_builder(
                root,
                implementation_commit,
                implementation_tree,
                allow_spec=delta.DEFAULT_ALLOW_SPEC,
                git_binary=_GIT_EXECUTABLE,
            )
        )
        validated = dict(
            delta_validator(
                root,
                delta.serialize_delta_manifest(recomputed),
                expected_implementation_commit=implementation_commit,
                expected_implementation_tree=implementation_tree,
                allow_spec=delta.DEFAULT_ALLOW_SPEC,
                git_binary=_GIT_EXECUTABLE,
            )
        )
    except Exception as exc:
        raise SecGemmaLeanV32PreflightError("implementation_delta_invalid") from exc
    if recomputed != validated or recomputed != repository["implementation_delta_manifest"]:
        raise SecGemmaLeanV32PreflightError("implementation_delta_invalid")
    if source_verifier is None:
        current_source = _verify_execution_closure(root, evidence_head, git_runner=git_runner)
    else:
        current_source = dict(source_verifier(root, evidence_head))
    receipt_source = repository["executed_source_identity"]
    if (
        current_source.get("source_sha256s") != receipt_source.get("source_sha256s")
        or current_source.get("source_inventory_sha256")
        != receipt_source.get("source_inventory_sha256")
    ):
        raise SecGemmaLeanV32PreflightError("executed_source_invalid")
    contact = contact_verifier(root)
    if contact != report["private_sec_contact"]["sha256"]:
        raise SecGemmaLeanV32PreflightError("private_contact_changed")
    marker_state = dict(marker_verifier(root, report))
    publish_state = dict(publish_verifier(root, report))
    checkpoint_root = root / Path(CHECKPOINT_ROOT)
    _assert_no_active_run(checkpoint_root)
    middle_snapshot = current_snapshot()
    final_contact = contact_verifier(root)
    final_marker_state = dict(marker_verifier(root, report))
    final_publish_state = dict(publish_verifier(root, report))
    _assert_no_active_run(checkpoint_root)
    final_snapshot = current_snapshot()
    if (
        middle_snapshot != initial_snapshot
        or final_snapshot != initial_snapshot
        or final_contact != contact
        or final_marker_state != marker_state
        or final_publish_state != publish_state
    ):
        raise SecGemmaLeanV32PreflightError("repository_changed_during_validation")
    return {
        "schema_version": "aapl-sec-gemma-lean-evidence-v3-2-acquisition-gate-v1",
        "validated": True,
        "read_only_validation": True,
        "evidence_commit": evidence_head,
        "evidence_tree": evidence_tree,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "preflight_sha256": report["preflight_sha256"],
        "implementation_delta_manifest_sha256": repository[
            "implementation_delta_manifest_sha256"
        ],
        "contact_sha256": contact,
        "durable_probe_state": {**marker_state, "publish_complete": publish_state},
        "artifact_only_evidence_commit": True,
        "raw_artifact_equals_committed_blob": True,
        "current_branch_upstream_clean": True,
    }


def validate_committed_preflight_for_acquisition(repo_root: Path) -> dict[str, Any]:
    """Strict read-only gate for acquisition after PREFLIGHT.json is committed."""

    return _validate_committed_preflight_for_acquisition(repo_root)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _serialize_preflight_report(report: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(report),
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise SecGemmaLeanV32PreflightError("receipt_not_canonical") from exc


def _publish_complete_body(report: Mapping[str, Any]) -> dict[str, Any]:
    repository = report.get("repository", {})
    checkpoint = report.get("checkpoint", {})
    durable = checkpoint.get("durable_identity_probe_evidence", {})
    try:
        intent_hash = durable["intent"]["marker_sha256"]
        completion_hash = durable["completion"]["marker_sha256"]
    except (KeyError, TypeError):
        raise SecGemmaLeanV32PreflightError("publish_state_invalid") from None
    return {
        "schema_version": "aapl-sec-gemma-lean-evidence-v3-2-publish-complete-v1",
        "preflight_sha256": report.get("preflight_sha256"),
        "artifact_path": PREFLIGHT_ARTIFACT_PATH,
        "artifact_file_sha256": hashlib.sha256(
            _serialize_preflight_report(report)
        ).hexdigest(),
        "implementation_commit": repository.get("head_commit"),
        "implementation_tree": repository.get("head_tree"),
        "intent_marker_sha256": intent_hash,
        "completion_marker_sha256": completion_hash,
    }


def _write_publish_complete(
    checkpoint_root: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return _write_durable_marker(
        checkpoint_root,
        PUBLISH_COMPLETE_NAME,
        _publish_complete_body(report),
    )


def _validate_publish_complete(
    repo_root: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    relative = f"{CHECKPOINT_ROOT}/{PUBLISH_COMPLETE_NAME}"
    payload = _safe_regular_bytes(
        repo_root,
        relative,
        maximum_bytes=64 * 1024,
        error_code="publish_state_invalid",
    )
    marker = _strict_json_object(payload, "publish_state_invalid")
    if payload != _canonical_json(marker) + b"\n":
        raise SecGemmaLeanV32PreflightError("publish_state_invalid")
    expected_keys = {
        "schema_version",
        "preflight_sha256",
        "artifact_path",
        "artifact_file_sha256",
        "implementation_commit",
        "implementation_tree",
        "intent_marker_sha256",
        "completion_marker_sha256",
        "marker_sha256",
    }
    body = {key: value for key, value in marker.items() if key != "marker_sha256"}
    if (
        set(marker) != expected_keys
        or marker.get("marker_sha256") != _canonical_sha256(body)
        or body != _publish_complete_body(report)
    ):
        raise SecGemmaLeanV32PreflightError("publish_state_invalid")
    return {
        "path": relative,
        "marker_sha256": marker["marker_sha256"],
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "artifact_file_sha256": marker["artifact_file_sha256"],
    }


def _atomic_publish(
    repo_root: Path,
    report: dict[str, Any],
    checkpoint_root: Path,
) -> str:
    report = _validate_sealable_report(report)
    relative = _relative_path(PREFLIGHT_ARTIFACT_PATH, suffix=".json")
    target = repo_root / Path(relative)
    _safe_directory(target.parent, repo_root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError("preflight_artifact_unwritable") from exc
    _safe_directory(target.parent, repo_root)
    if target.exists():
        raise SecGemmaLeanV32PreflightError("preflight_already_sealed")
    if checkpoint_root.absolute() != (repo_root / Path(CHECKPOINT_ROOT)).absolute():
        raise SecGemmaLeanV32PreflightError("checkpoint_root_invalid")
    try:
        if checkpoint_root.stat().st_dev != target.parent.stat().st_dev:
            raise SecGemmaLeanV32PreflightError("preflight_artifact_unwritable")
    except OSError as exc:
        raise SecGemmaLeanV32PreflightError("preflight_artifact_unwritable") from exc
    payload = _serialize_preflight_report(report)
    temporary = checkpoint_root / f".preflight-seal-{uuid.uuid4().hex}.tmp"
    renamed = False
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # The ignored checkpoint and tracked evidence directory are children
        # of the same repository filesystem.  The held lease excludes another
        # production preflight; the existence check above supplies no-replace
        # behavior for that cooperative writer set.
        os.rename(temporary, target)
        renamed = True
        _fsync_directory(target.parent)
        details = target.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or _is_reparse(details)
            or details.st_nlink != 1
            or target.read_bytes() != payload
        ):
            raise OSError("published bytes changed")
    except FileExistsError:
        if not renamed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise SecGemmaLeanV32PreflightError("preflight_already_sealed") from None
    except OSError:
        if not renamed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        # Once rename was attempted successfully, never delete the target.
        # A post-rename read/fsync error is uncertain publication, and leaving
        # the evidence plus durable probe markers is the fail-closed outcome.
        raise SecGemmaLeanV32PreflightError("preflight_artifact_unwritable") from None
    return relative


def run_preflight(repo_root: Path) -> dict[str, Any]:
    """Run the fixed production preflight.  No production run starts here."""

    root = _canonical_repo_root(repo_root)
    if (root / Path(PREFLIGHT_ARTIFACT_PATH)).exists():
        raise SecGemmaLeanV32PreflightError("preflight_already_sealed")
    checkpoint = _prepare_checkpoint_root(root)
    _assert_no_prior_identity_probe(checkpoint)
    _assert_no_active_run(checkpoint)
    started = time.monotonic()
    with _PreflightLease(checkpoint):
        before = _verify_repository(root)
        fingerprint = _verify_private_contact(root)
        runtime, probe_evidence = _durable_production_identity_probe(
            checkpoint,
            before,
            fingerprint,
        )
        after = _verify_repository(root)
        _assert_no_active_run(
            checkpoint,
            owned_preflight_lock=checkpoint / ACTIVE_LOCK_NAME,
        )
        _write_probe(checkpoint)
        finished = time.monotonic()
        report = _build_report(
            repository_before=before,
            repository_after=after,
            contact_fingerprint=fingerprint,
            runtime=runtime,
            created_at=_utc_now(),
            elapsed_seconds=finished - started,
            trusted=True,
            artifact_will_dirty=True,
            probe_evidence=probe_evidence,
        )
        _validate_sealable_report(report)
    # Publication occurs only after the cooperative lease has been removed
    # successfully.  No identity request is repeated during these rebound
    # checks.
    _assert_no_active_run(checkpoint)
    final_repository = _verify_repository(root)
    if _canonical_sha256(_stable_repository(final_repository)) != _canonical_sha256(
        _stable_repository(after)
    ):
        raise SecGemmaLeanV32PreflightError("repository_changed_during_preflight")
    if _verify_private_contact(root) != fingerprint:
        raise SecGemmaLeanV32PreflightError("private_contact_changed")
    _validate_durable_probe_state(root, report)
    _assert_no_active_run(checkpoint)
    _atomic_publish(root, report, checkpoint)
    _write_publish_complete(checkpoint, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="SEC/Gemma lean v3.2 zero-effect preflight")
    parser.add_argument("repo_root", nargs="?", default=".")
    arguments = parser.parse_args(argv)
    try:
        report = run_preflight(Path(arguments.repo_root))
    except SecGemmaLeanV32PreflightError as exc:
        print(json.dumps({"status": "failed", "code": exc.code}, sort_keys=True))
        return 1
    print(json.dumps({"status": report["status"], "preflight_sha256": report["preflight_sha256"]}, sort_keys=True))
    return 0


__all__ = [
    "ALLOWED_IDENTITY_SEQUENCE",
    "BRANCH_NAME",
    "CHECKPOINT_ROOT",
    "EXECUTED_SOURCE_CLOSURE",
    "MODEL_MANIFEST_SHA256",
    "MODEL_NAME",
    "OLLAMA_VERSION_CANONICAL_BYTES",
    "PREFLIGHT_ARTIFACT_PATH",
    "PRIVATE_CONFIG_PATH",
    "SCHEMA_VERSION",
    "SecGemmaLeanV32PreflightError",
    "run_preflight",
    "validate_committed_preflight_for_acquisition",
]


if __name__ == "__main__":
    raise SystemExit(main())
