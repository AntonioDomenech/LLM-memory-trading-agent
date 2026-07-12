"""Effectful, local persistence for SEC/Gemma reveal governance.

The sibling :mod:`sec_filing_gemma_reveal_registry` module deliberately has no
I/O.  This module is its narrow effectful boundary.  It authenticates the
checked-in genesis evidence, keeps one authoritative registry/pin pair, applies
registry appends with compare-and-swap semantics, and consumes stage reveal
requests exactly once only after the fixed semantic prerequisite verifier
returns a strongly bound result.

No market data, filing text, model runtime, clock, or network service is used.
The store contains governance receipts only.  Registration and intermediate
access never count as a final-period touch; successful final-request
consumption increments the separate actual-final-touch counter exactly once.
"""

from __future__ import annotations

from collections.abc import Mapping
import base64
import copy
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
import time
from types import MappingProxyType
from typing import Any, Final
import uuid

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    SecFilingGemmaContractError,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND,
    HISTORICAL_REGISTRY_CANONICAL_SHA256,
    REVEAL_REQUEST_SCHEMA_VERSION,
    SecFilingGemmaRevealRegistryError,
    build_initial_reveal_registry,
    derive_registry_pin,
    historical_reveal_declaration,
    validate_registry_pin_transition,
    validate_reveal_registry,
    validate_single_candidate_reveal_request,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    STAGE_AUDIT_RECEIPT_SCHEMA_VERSION,
    authoritative_prerequisite_validator,
    detach_untrusted_stage_json,
    preflight_untrusted_stage_json,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
    SecFilingGemmaStageAuthorizationError,
    authenticate_reveal_store_trusted_stage_content_pin,
    build_reveal_store_current_tip_anchor,
    build_consumed_stage_authorization_grant,
    build_consumed_stage_output_receipt,
    derive_consumed_stage_store_state_pin,
    derive_reveal_store_trusted_stage_content_pin,
    validate_consumed_stage_authorization_grant,
    validate_consumed_stage_output_receipt,
    validate_reveal_store_current_tip_anchor,
    validate_reveal_store_current_tip_anchor_structure,
    validate_reveal_store_current_tip_anchor_transition,
    validate_trusted_stage_content_authentication_receipt,
)


STORE_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-reveal-store-v1"
STORE_ANCHOR_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-reveal-store-anchor-v1"
CONSUMPTION_LEDGER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-request-ledger-v1"
)
CONSUMPTION_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-request-entry-v1"
)
SEMANTIC_PREREQUISITE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-semantic-prerequisite-validation-v1"
)
INITIAL_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-initial-external-registry-pin-v1"
)
STATE_FILENAME: Final[str] = "sec_gemma_reveal_store.json"
CURRENT_TIP_ANCHOR_FILENAME: Final[str] = (
    "sec_gemma_reveal_store_current_tip.json"
)
RESTORE_PENDING_FILENAME: Final[str] = (
    "sec_gemma_reveal_store_restore_pending.json"
)
LOCK_FILENAME: Final[str] = ".sec_gemma_reveal_store.lock"
INITIAL_PIN_RELATIVE_PATH: Final[str] = (
    "docs/protocol_evidence/sec_gemma_reveal_registry_initial_pin.json"
)
REQUIRED_SEMANTIC_CHECKS: Final[tuple[str, ...]] = (
    REQUIRED_STAGE_VERIFIER_CHECKS
)
AUTHORITATIVE_VALIDATOR_ID: Final[str] = (
    "aapl-sec-gemma-authoritative-stage-verifier-v1"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
CURRENT_TIP_PENDING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-reveal-store-current-tip-pending-v1"
)
RESTORE_PENDING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-reveal-store-restore-pending-v1"
)
AUTHORITATIVE_STAGE_PROMOTION_ENABLED: Final[bool] = False
AUTHENTICATED_STORE_VERIFIER_CONTEXT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-authenticated-store-verifier-context-v1"
)
PARENT_CONSUMPTION_BINDING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-parent-consumption-binding-v2"
)
MAX_STATE_FILE_BYTES: Final[int] = 16 * 1024 * 1024
MAX_CURRENT_TIP_ANCHOR_FILE_BYTES: Final[int] = 64 * 1024 * 1024
MAX_TRACKED_ANCHOR_FILE_BYTES: Final[int] = 8 * 1024 * 1024
MAX_RESTORE_PENDING_FILE_BYTES: Final[int] = 128 * 1024 * 1024
_TEMP_RE = re.compile(
    rf"\.(?:{re.escape(STATE_FILENAME)}|"
    rf"{re.escape(CURRENT_TIP_ANCHOR_FILENAME)}|"
    rf"{re.escape(RESTORE_PENDING_FILENAME)})\.[0-9a-f]{{32}}\.tmp\Z"
)
_WINDOWS_REPARSE_POINT = 0x400
_BINARY = getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "registry_sha256",
        "registry_tip_sha256",
        "registered_entry_count",
        "historical_final_reveal_count_lower_bound",
        "stage",
        "stage_access_manifest_sha256",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_scope",
        "authorizes_outcome_access",
        "effectful_atomic_single_use_consumption_required",
        "cross_attempt_comparison_permitted",
        "cross_attempt_winner_selection_permitted",
        "globally_pristine_claim",
        "request_sha256",
    }
)


class SecFilingGemmaRevealStoreError(ValueError):
    """The local reveal store failed closed on unsafe or inconsistent state."""


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SecFilingGemmaRevealStoreError(f"{location} must be a JSON object")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaRevealStoreError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be an integer >= {minimum}"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _safe_id(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise SecFilingGemmaRevealStoreError(f"{location} is not a safe identity")
    return value


def _semantic_checks(value: Any, location: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be the exact frozen verifier checklist"
        )
    normalized = tuple(value)
    if normalized != REQUIRED_SEMANTIC_CHECKS:
        raise SecFilingGemmaRevealStoreError(
            f"{location} does not match the exact frozen verifier checklist"
        )
    return normalized


def _same_digest(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def _json_value_copy(value: Any, location: str) -> Any:
    """Copy one finite JSON value and reject non-JSON or duplicate-key hazards."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be a finite JSON value"
        ) from exc


def _exact_builtin_json_copy(value: Any, location: str) -> Any:
    """Detach an authorization-critical caller value without invoking hooks.

    Only exact built-in containers and scalar types are traversed.  In
    particular, a ``Mapping`` implementation or a ``dict`` subclass is
    rejected before any caller-controlled iterator, item accessor, encoder,
    comparison, or representation method can execute.
    """

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SecFilingGemmaRevealStoreError(
                f"{location} contains a non-finite float"
            )
        return value
    if type(value) is list:
        return [
            _exact_builtin_json_copy(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if type(value) is tuple:
        return [
            _exact_builtin_json_copy(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if type(value) is dict:
        detached: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise SecFilingGemmaRevealStoreError(
                    f"{location} contains a non-string key"
                )
            detached[key] = _exact_builtin_json_copy(
                child, f"{location}.{key}"
            )
        return detached
    raise SecFilingGemmaRevealStoreError(
        f"{location} must contain only exact built-in JSON values"
    )


def _exact_caller_dict(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be an exact built-in dict"
        )
    detached = _exact_builtin_json_copy(value, location)
    if type(detached) is not dict:  # pragma: no cover - guaranteed above
        raise SecFilingGemmaRevealStoreError(f"{location} must be an object")
    return detached


def _strict_json_bytes(payload: bytes, location: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecFilingGemmaRevealStoreError(
                    f"{location} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise SecFilingGemmaRevealStoreError(
            f"{location} contains non-finite JSON constant {value}"
        )

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except SecFilingGemmaRevealStoreError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SecFilingGemmaRevealStoreError(
            f"{location} is not strict UTF-8 JSON"
        ) from exc


def _encoded_state(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store state is not finite JSON"
        ) from exc


def _is_reparse(details: os.stat_result) -> bool:
    return bool(
        getattr(details, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT
    )


def _validate_real_directory(path: Path, location: str) -> os.stat_result:
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise SecFilingGemmaRevealStoreError(f"{location} does not exist") from exc
    if stat.S_ISLNK(details.st_mode) or _is_reparse(details):
        raise SecFilingGemmaRevealStoreError(
            f"{location} cannot be a link or reparse point"
        )
    if not stat.S_ISDIR(details.st_mode):
        raise SecFilingGemmaRevealStoreError(f"{location} must be a directory")
    return details


def _secure_directory(path: Path, *, create: bool, location: str) -> Path:
    raw = os.fspath(path)
    absolute = Path(os.path.abspath(raw))
    windows = PureWindowsPath(str(absolute))
    if str(absolute).startswith(("\\\\", "//")) or windows.drive.startswith("\\"):
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be on a local filesystem"
        )
    if not absolute.anchor:
        raise SecFilingGemmaRevealStoreError(f"{location} must be absolute")

    anchor = Path(absolute.anchor)
    _validate_real_directory(anchor, f"{location} filesystem root")
    current = anchor
    for component in absolute.relative_to(anchor).parts:
        if component in {"", ".", ".."}:
            raise SecFilingGemmaRevealStoreError(
                f"{location} contains an unsafe path component"
            )
        current = current / component
        try:
            _validate_real_directory(current, location)
        except SecFilingGemmaRevealStoreError as exc:
            if not create or current.exists() or current.is_symlink():
                raise
            try:
                current.mkdir()
            except FileExistsError:
                pass
            _validate_real_directory(current, location)
    return absolute


def _validate_regular_details(details: os.stat_result, location: str) -> None:
    if (
        stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
    ):
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be a single-link regular file, never a link or "
            "reparse point"
        )


def _read_regular_bytes(
    path: Path,
    location: str,
    *,
    max_bytes: int = MAX_TRACKED_ANCHOR_FILE_BYTES,
) -> bytes:
    if type(max_bytes) is not int or max_bytes < 1:
        raise SecFilingGemmaRevealStoreError("Regular-file read cap is invalid")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise SecFilingGemmaRevealStoreError(f"{location} does not exist") from exc
    _validate_regular_details(before, location)
    if before.st_size > max_bytes:
        raise SecFilingGemmaRevealStoreError(
            f"{location} exceeds the {max_bytes}-byte safety limit"
        )
    flags = os.O_RDONLY | _BINARY | _NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SecFilingGemmaRevealStoreError(
            f"Could not securely open {location}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        _validate_regular_details(opened, location)
        if opened.st_size > max_bytes:
            raise SecFilingGemmaRevealStoreError(
                f"{location} exceeds the {max_bytes}-byte safety limit"
            )
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise SecFilingGemmaRevealStoreError(
                f"{location} identity changed while opening"
            )
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed_size))
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > max_bytes:
                raise SecFilingGemmaRevealStoreError(
                    f"{location} exceeded the {max_bytes}-byte safety limit while reading"
                )
            chunks.append(chunk)
        after = path.lstat()
        _validate_regular_details(after, location)
        if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
            raise SecFilingGemmaRevealStoreError(
                f"{location} identity changed while reading"
            )
        if after.st_size > max_bytes or after.st_size != observed_size:
            raise SecFilingGemmaRevealStoreError(
                f"{location} size changed or exceeded its limit while reading"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _git(repo_root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SecFilingGemmaRevealStoreError(
            "Could not authenticate reveal-store anchors through local Git"
        ) from exc
    if completed.returncode != 0:
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store anchors must exist in the current Git HEAD"
        )
    return completed.stdout


def _tracked_head_bytes(repo_root: Path, relative_name: str) -> bytes:
    relative = PurePosixPath(relative_name)
    windows = PureWindowsPath(relative_name)
    if (
        relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in relative.parts
        or len(relative.parts) < 1
    ):
        raise SecFilingGemmaRevealStoreError("Tracked anchor path is unsafe")

    top_level_raw = _git(repo_root, "rev-parse", "--show-toplevel").strip()
    try:
        top_level = Path(os.path.abspath(os.fsdecode(top_level_raw)))
    except UnicodeError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Git repository root is not a valid local path"
        ) from exc
    if os.path.normcase(str(top_level)) != os.path.normcase(str(repo_root)):
        raise SecFilingGemmaRevealStoreError(
            "repository_root must be the exact Git worktree root"
        )

    blob = _git(repo_root, "cat-file", "blob", f"HEAD:{relative.as_posix()}")
    worktree_path = repo_root.joinpath(*relative.parts)
    observed = _read_regular_bytes(
        worktree_path, f"tracked anchor {relative.as_posix()}"
    )
    # Git may materialize tracked JSON with platform line endings.  Authenticate
    # the strict JSON value rather than requiring byte-identical checkout EOLs.
    try:
        observed_value = _strict_json_bytes(
            observed, f"tracked anchor {relative.as_posix()}"
        )
        head_value = _strict_json_bytes(
            blob, f"Git HEAD anchor {relative.as_posix()}"
        )
        same_value = _same_digest(
            canonical_sha256(observed_value), canonical_sha256(head_value)
        )
    except SecFilingGemmaContractError as exc:
        raise SecFilingGemmaRevealStoreError(
            f"Tracked anchor {relative.as_posix()} is not finite canonical JSON"
        ) from exc
    if not same_value:
        raise SecFilingGemmaRevealStoreError(
            f"Tracked anchor {relative.as_posix()} does not match Git HEAD"
        )
    return blob


def _load_tracked_anchor(repo_root: Path) -> dict[str, Any]:
    pin_payload = _strict_json_bytes(
        _tracked_head_bytes(repo_root, INITIAL_PIN_RELATIVE_PATH),
        "initial registry-pin artifact",
    )
    pin = _expect_mapping(pin_payload, "initial registry-pin artifact")
    _expect_keys(
        pin,
        {
            "schema_version",
            "contract_version",
            "migration_source_canonical_sha256",
            "registry_pin",
        },
        "initial registry-pin artifact",
    )
    if pin["schema_version"] != INITIAL_PIN_SCHEMA_VERSION:
        raise SecFilingGemmaRevealStoreError("Unknown initial registry-pin schema")
    if pin["contract_version"] != CONTRACT_VERSION:
        raise SecFilingGemmaRevealStoreError(
            "Initial registry pin belongs to another experiment contract"
        )

    declaration = historical_reveal_declaration()
    migration_name = declaration["migration_source"]
    if not isinstance(migration_name, str):
        raise SecFilingGemmaRevealStoreError("Migration source path is invalid")
    migration_payload = _strict_json_bytes(
        _tracked_head_bytes(repo_root, migration_name),
        "historical migration snapshot",
    )
    try:
        migration_hash = canonical_sha256(migration_payload)
    except SecFilingGemmaContractError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Historical migration snapshot is not canonical finite JSON"
        ) from exc
    expected_migration_hash = declaration["migration_source_canonical_sha256"]
    if (
        not isinstance(expected_migration_hash, str)
        or not _same_digest(expected_migration_hash, HISTORICAL_REGISTRY_CANONICAL_SHA256)
        or not _same_digest(migration_hash, expected_migration_hash)
        or pin["migration_source_canonical_sha256"] != expected_migration_hash
    ):
        raise SecFilingGemmaRevealStoreError(
            "Tracked migration snapshot canonical hash does not match the frozen anchor"
        )

    initial_registry = build_initial_reveal_registry()
    expected_pin = derive_registry_pin(initial_registry)
    if pin["registry_pin"] != expected_pin:
        raise SecFilingGemmaRevealStoreError(
            "Tracked initial pin does not authenticate the deterministic genesis registry"
        )
    pin_copy = _json_value_copy(dict(pin), "initial registry-pin artifact")
    return {
        "schema_version": STORE_ANCHOR_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "initial_pin_relative_path": INITIAL_PIN_RELATIVE_PATH,
        "initial_pin_artifact_canonical_sha256": canonical_sha256(pin_copy),
        "migration_source": migration_name,
        "migration_source_canonical_sha256": migration_hash,
        "initial_registry_sha256": initial_registry["registry_sha256"],
        "initial_registry_pin": expected_pin,
        "historical_final_reveal_count_lower_bound": (
            HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
        ),
    }


class _ExclusiveFileLock:
    def __init__(self, path: Path, *, timeout_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.descriptor: int | None = None

    def __enter__(self) -> "_ExclusiveFileLock":
        flags = os.O_RDWR | os.O_CREAT | _BINARY | _NOFOLLOW
        before: os.stat_result | None = None
        try:
            before = self.path.lstat()
            _validate_regular_details(before, "reveal-store lock")
        except FileNotFoundError:
            pass
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Could not securely open the reveal-store lock"
            ) from exc
        try:
            details = os.fstat(descriptor)
            _validate_regular_details(details, "reveal-store lock")
            after = self.path.lstat()
            _validate_regular_details(after, "reveal-store lock")
            if (
                (details.st_dev, details.st_ino) != (after.st_dev, after.st_ino)
                or before is not None
                and (before.st_dev, before.st_ino)
                != (details.st_dev, details.st_ino)
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Reveal-store lock identity changed while opening"
                )
            if details.st_size == 0:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise SecFilingGemmaRevealStoreError(
                            "Timed out acquiring the exclusive reveal-store lock"
                        ) from exc
                    time.sleep(0.025)
            self.descriptor = descriptor
            return self
        except Exception:
            os.close(descriptor)
            raise

    def __exit__(self, *args: Any) -> None:
        descriptor, self.descriptor = self.descriptor, None
        if descriptor is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | _BINARY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        # Windows commonly refuses directory fsync through the CRT.  The file
        # itself is always fsynced before replace; directory fsync is best effort.
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise SecFilingGemmaRevealStoreError(
                    "Could not finish writing reveal-store temporary state"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        if path.exists() or path.is_symlink():
            _validate_regular_details(path.lstat(), "authoritative reveal-store state")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            details = temporary.lstat()
        except FileNotFoundError:
            return
        _validate_regular_details(details, "reveal-store temporary state")
        temporary.unlink()


def _cleanup_interrupted_temporaries(directory: Path) -> None:
    for entry in os.scandir(directory):
        if _TEMP_RE.fullmatch(entry.name) is None:
            continue
        path = Path(entry.path)
        details = path.lstat()
        _validate_regular_details(details, "interrupted reveal-store temporary state")
        path.unlink()
    _fsync_directory(directory)


def _consumption_genesis(anchor: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "aapl-sec-gemma-consumed-request-genesis-v1",
            "contract_version": CONTRACT_VERSION,
            "store_anchor_sha256": canonical_sha256(anchor),
        }
    )


def _consumption_ledger(
    entries: list[dict[str, Any]], *, tip_sha256: str, anchor: Mapping[str, Any]
) -> dict[str, Any]:
    actual_final_touch_count = sum(
        1 for entry in entries if entry["stage"] == "final"
    )
    body = {
        "schema_version": CONSUMPTION_LEDGER_SCHEMA_VERSION,
        "entries": copy.deepcopy(entries),
        "chain": {
            "genesis_tip_sha256": _consumption_genesis(anchor),
            "tip_sha256": tip_sha256,
            "consumed_request_count": len(entries),
            "actual_final_touch_count": actual_final_touch_count,
            "historical_final_reveal_count_lower_bound": (
                HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
            ),
            "repository_final_touch_count_lower_bound": (
                HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
                + actual_final_touch_count
            ),
        },
    }
    return {**body, "ledger_sha256": canonical_sha256(body)}


def _initial_consumption_ledger(anchor: Mapping[str, Any]) -> dict[str, Any]:
    genesis = _consumption_genesis(anchor)
    return _consumption_ledger([], tip_sha256=genesis, anchor=anchor)


def _state_snapshot(
    *,
    anchor: Mapping[str, Any],
    latest_registry: Mapping[str, Any],
    latest_pin: Mapping[str, Any],
    consumption_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": STORE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "anchor": copy.deepcopy(dict(anchor)),
        "latest_registry": copy.deepcopy(dict(latest_registry)),
        "latest_registry_pin": copy.deepcopy(dict(latest_pin)),
        "consumption_ledger": copy.deepcopy(dict(consumption_ledger)),
    }
    return {**body, "state_sha256": canonical_sha256(body)}


def _pending_current_tip_document(
    *,
    prior_tip_anchor: Mapping[str, Any] | None,
    next_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    prior = (
        None
        if prior_tip_anchor is None
        else _exact_caller_dict(dict(prior_tip_anchor), "prior current-tip anchor")
    )
    next_anchor = _exact_caller_dict(
        dict(next_tip_anchor), "next current-tip anchor"
    )
    if prior is None:
        validated_next = validate_reveal_store_current_tip_anchor_structure(
            next_anchor
        )
        if (
            validated_next["revision"] != 0
            or validated_next["previous_tip_anchor_sha256"] is not None
        ):
            raise SecFilingGemmaRevealStoreError(
                "Only the deterministic genesis may omit a prior current-tip anchor"
            )
    else:
        try:
            prior, validated_next = validate_reveal_store_current_tip_anchor_transition(
                prior, next_anchor
            )
        except SecFilingGemmaStageAuthorizationError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Current-tip transaction is not a valid monotonic CAS transition"
            ) from exc
    body = {
        "schema_version": CURRENT_TIP_PENDING_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "prior_tip_anchor": prior,
        "next_tip_anchor": validated_next,
    }
    return {**body, "pending_sha256": canonical_sha256(body)}


def _validated_pending_current_tip_document(raw: Any) -> dict[str, Any]:
    try:
        detached = detach_untrusted_stage_json(
            raw, "pending current-tip transaction"
        )
    except Exception as exc:
        raise SecFilingGemmaRevealStoreError(
            "Pending current-tip transaction exceeds fixed allocation bounds"
        ) from exc
    value = _expect_mapping(detached, "pending current-tip transaction")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_version",
            "prior_tip_anchor",
            "next_tip_anchor",
            "pending_sha256",
        },
        "pending current-tip transaction",
    )
    if (
        value["schema_version"] != CURRENT_TIP_PENDING_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaRevealStoreError(
            "Pending current-tip transaction schema or contract changed"
        )
    observed = _sha256(value["pending_sha256"], "pending current-tip hash")
    body = {key: value[key] for key in value if key != "pending_sha256"}
    if not _same_digest(observed, canonical_sha256(body)):
        raise SecFilingGemmaRevealStoreError(
            "Pending current-tip transaction hash is inconsistent"
        )
    try:
        prior_raw = value["prior_tip_anchor"]
        if prior_raw is None:
            next_anchor = validate_reveal_store_current_tip_anchor_structure(
                value["next_tip_anchor"]
            )
            if (
                next_anchor["revision"] != 0
                or next_anchor["previous_tip_anchor_sha256"] is not None
            ):
                raise SecFilingGemmaStageAuthorizationError(
                    "Non-genesis pending anchor has no predecessor"
                )
            prior = None
        else:
            prior, next_anchor = validate_reveal_store_current_tip_anchor_transition(
                prior_raw,
                value["next_tip_anchor"],
            )
    except SecFilingGemmaStageAuthorizationError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Pending current-tip transaction is stale, forked, or invalid"
        ) from exc
    return {
        **body,
        "prior_tip_anchor": prior,
        "next_tip_anchor": next_anchor,
        "pending_sha256": observed,
    }


def _restore_pending_document(
    *,
    state_bytes: bytes,
    tip_anchor_bytes: bytes,
) -> dict[str, Any]:
    if (
        type(state_bytes) is not bytes
        or not state_bytes
        or len(state_bytes) > MAX_STATE_FILE_BYTES
        or type(tip_anchor_bytes) is not bytes
        or not tip_anchor_bytes
        or len(tip_anchor_bytes) > MAX_CURRENT_TIP_ANCHOR_FILE_BYTES
    ):
        raise SecFilingGemmaRevealStoreError(
            "Restore transaction target bytes exceed their safety limits"
        )
    body = {
        "schema_version": RESTORE_PENDING_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "state_bytes_base64": base64.b64encode(state_bytes).decode("ascii"),
        "state_bytes_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "state_byte_count": len(state_bytes),
        "tip_anchor_bytes_base64": base64.b64encode(tip_anchor_bytes).decode(
            "ascii"
        ),
        "tip_anchor_bytes_sha256": hashlib.sha256(
            tip_anchor_bytes
        ).hexdigest(),
        "tip_anchor_byte_count": len(tip_anchor_bytes),
    }
    document = {**body, "restore_pending_sha256": canonical_sha256(body)}
    if len(_encoded_state(document)) > MAX_RESTORE_PENDING_FILE_BYTES:
        raise SecFilingGemmaRevealStoreError(
            "Restore transaction document exceeds its safety limit"
        )
    return document


def _decode_restore_bytes(
    encoded: Any,
    *,
    expected_sha256: Any,
    expected_byte_count: Any,
    maximum_bytes: int,
    location: str,
) -> bytes:
    if type(encoded) is not str or not encoded:
        raise SecFilingGemmaRevealStoreError(
            f"{location} must be nonempty canonical Base64"
        )
    expected_hash = _sha256(expected_sha256, f"{location} hash")
    byte_count = _strict_int(
        expected_byte_count, f"{location} byte count", minimum=1
    )
    if byte_count > maximum_bytes:
        raise SecFilingGemmaRevealStoreError(
            f"{location} exceeds its safety limit"
        )
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SecFilingGemmaRevealStoreError(
            f"{location} is not strict Base64"
        ) from exc
    if (
        len(payload) != byte_count
        or base64.b64encode(payload).decode("ascii") != encoded
        or not _same_digest(hashlib.sha256(payload).hexdigest(), expected_hash)
    ):
        raise SecFilingGemmaRevealStoreError(
            f"{location} bytes do not match their exact recovery pin"
        )
    return payload


def _validated_restore_pending_document(
    raw: Any,
    *,
    tracked_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        detached = detach_untrusted_stage_json(
            raw, "reveal-store restore transaction"
        )
    except Exception as exc:
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store restore transaction exceeds fixed allocation bounds"
        ) from exc
    value = _expect_mapping(detached, "reveal-store restore transaction")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_version",
            "state_bytes_base64",
            "state_bytes_sha256",
            "state_byte_count",
            "tip_anchor_bytes_base64",
            "tip_anchor_bytes_sha256",
            "tip_anchor_byte_count",
            "restore_pending_sha256",
        },
        "reveal-store restore transaction",
    )
    if (
        value["schema_version"] != RESTORE_PENDING_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store restore transaction schema or contract changed"
        )
    observed = _sha256(
        value["restore_pending_sha256"], "restore transaction self-hash"
    )
    body = {
        key: value[key]
        for key in value
        if key != "restore_pending_sha256"
    }
    if not _same_digest(observed, canonical_sha256(body)):
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store restore transaction self-hash is inconsistent"
        )
    state_bytes = _decode_restore_bytes(
        value["state_bytes_base64"],
        expected_sha256=value["state_bytes_sha256"],
        expected_byte_count=value["state_byte_count"],
        maximum_bytes=MAX_STATE_FILE_BYTES,
        location="restore target state",
    )
    tip_bytes = _decode_restore_bytes(
        value["tip_anchor_bytes_base64"],
        expected_sha256=value["tip_anchor_bytes_sha256"],
        expected_byte_count=value["tip_anchor_byte_count"],
        maximum_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
        location="restore target current-tip anchor",
    )
    try:
        target_state = detach_untrusted_stage_json(
            _strict_json_bytes(state_bytes, "restore target state"),
            "restore target state",
        )
        target_tip = detach_untrusted_stage_json(
            _strict_json_bytes(tip_bytes, "restore target current-tip anchor"),
            "restore target current-tip anchor",
        )
    except Exception as exc:
        raise SecFilingGemmaRevealStoreError(
            "Restore targets exceed fixed allocation bounds"
        ) from exc
    state = _validate_state(
        _expect_mapping(target_state, "restore target state"),
        expected_anchor=tracked_anchor,
    )
    tip_mapping = _expect_mapping(
        target_tip,
        "restore target current-tip anchor",
    )
    try:
        tip = validate_reveal_store_current_tip_anchor(state, tip_mapping)
    except SecFilingGemmaStageAuthorizationError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Restore target tip does not authenticate its target state"
        ) from exc
    return {
        "state": state,
        "tip_anchor": tip,
        "state_bytes": state_bytes,
        "tip_anchor_bytes": tip_bytes,
        "restore_pending_sha256": observed,
    }


def _state_bytes_match_tip_anchor(
    state_bytes: bytes, state: Mapping[str, Any], tip_anchor: Mapping[str, Any]
) -> bool:
    return (
        tip_anchor["state_sha256"] == state.get("state_sha256")
        and tip_anchor["state_snapshot_byte_count"] == len(state_bytes)
        and _same_digest(
            tip_anchor["state_snapshot_bytes_sha256"],
            hashlib.sha256(state_bytes).hexdigest(),
        )
    )


def _authorization_bundle(
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    expected_new_consumption_entry_sha256: str,
) -> dict[str, Any]:
    snapshot = _exact_caller_dict(
        dict(authenticated_store_snapshot),
        "authenticated post-consumption store snapshot",
    )
    state_bytes = _encoded_state(snapshot)
    store_pin = derive_consumed_stage_store_state_pin(snapshot)
    grant = build_consumed_stage_authorization_grant(
        authenticated_store_snapshot=snapshot,
        external_store_state_pin=store_pin,
        expected_new_consumption_entry_sha256=(
            expected_new_consumption_entry_sha256
        ),
    )
    if (
        grant["store_snapshot_bytes_sha256"]
        != hashlib.sha256(state_bytes).hexdigest()
        or grant["store_snapshot_byte_count"] != len(state_bytes)
    ):
        raise SecFilingGemmaRevealStoreError(
            "Authorization grant does not bind the exact post-consumption bytes"
        )
    body = {
        "schema_version": CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
        "authenticated_store_snapshot": snapshot,
        "store_state_pin": store_pin,
        "authorization_grant": grant,
    }
    return {**body, "bundle_sha256": canonical_sha256(body)}


def _authenticated_store_verifier_context(
    *,
    trusted_stage_content_pin: Mapping[str, Any],
    trusted_stage_content_authentication: Mapping[str, Any],
    parent_consumption_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    pin = _exact_caller_dict(
        dict(trusted_stage_content_pin),
        "authenticated verifier trusted content pin",
    )
    authentication = _exact_caller_dict(
        dict(trusted_stage_content_authentication),
        "authenticated verifier trusted content receipt",
    )
    parent = (
        None
        if parent_consumption_binding is None
        else _exact_caller_dict(
            dict(parent_consumption_binding),
            "authenticated verifier parent-consumption binding",
        )
    )
    body = {
        "schema_version": AUTHENTICATED_STORE_VERIFIER_CONTEXT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "context_kind": "reveal_store_authenticated_preconsumption_context",
        "trusted_stage_content_pin": pin,
        "trusted_stage_content_authentication": authentication,
        "parent_consumption_binding": parent,
    }
    return {**body, "authenticated_store_context_sha256": canonical_sha256(body)}


def _request_body_hash(request: Mapping[str, Any]) -> str:
    _expect_keys(request, set(_REQUEST_KEYS), "stored reveal request")
    observed = _sha256(request["request_sha256"], "stored request_sha256")
    body = {key: request[key] for key in request if key != "request_sha256"}
    expected = canonical_sha256(body)
    if not _same_digest(observed, expected):
        raise SecFilingGemmaRevealStoreError(
            "Stored reveal request is not its canonical hashed body"
        )
    if (
        request["schema_version"] != REVEAL_REQUEST_SCHEMA_VERSION
        or request["contract_version"] != CONTRACT_VERSION
        or request["stage"] not in {"intermediate", "final"}
        or request["authorizes_outcome_access"] is not False
        or request["effectful_atomic_single_use_consumption_required"] is not True
        or request["cross_attempt_comparison_permitted"] is not False
        or request["cross_attempt_winner_selection_permitted"] is not False
        or request["globally_pristine_claim"] is not False
    ):
        raise SecFilingGemmaRevealStoreError(
            "Stored request changed its non-authorizing single-candidate semantics"
        )
    return expected


@dataclass(frozen=True)
class SemanticPrerequisiteValidation:
    """Strongly bound result returned by an independent semantic validator.

    A truthy value or ad-hoc mapping is intentionally insufficient.  The
    caller must separately pin ``validator_id`` and ``validator_source_sha256``;
    the store verifies both and all stage/candidate/evidence bindings.
    """

    validator_id: str
    validator_source_sha256: str
    prerequisite_stage: str
    prerequisite_stage_evidence_sha256: str
    attempt_id: str
    candidate_sha256: str
    candidate_design_sha256: str
    registry_entry_sha256: str
    request_sha256: str
    requested_stage: str
    stage_access_manifest_sha256: str
    registry_sha256: str
    registry_tip_sha256: str
    semantic_checks: tuple[str, ...]
    semantic_receipt: Mapping[str, Any]

    @classmethod
    def success(
        cls,
        expected_context: Mapping[str, Any],
        *,
        validator_id: str,
        validator_source_sha256: str,
        semantic_checks: tuple[str, ...],
        semantic_receipt: Mapping[str, Any],
    ) -> "SemanticPrerequisiteValidation":
        context = _expect_mapping(expected_context, "prerequisite context")
        required = {
            "prerequisite_stage",
            "prerequisite_stage_evidence_sha256",
            "attempt_id",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
            "request_sha256",
            "stage",
            "stage_access_manifest_sha256",
            "registry_sha256",
            "registry_tip_sha256",
        }
        if not required <= set(context):
            raise SecFilingGemmaRevealStoreError(
                "Prerequisite context is missing immutable request bindings"
            )
        return cls(
            validator_id=validator_id,
            validator_source_sha256=validator_source_sha256,
            prerequisite_stage=context["prerequisite_stage"],
            prerequisite_stage_evidence_sha256=context[
                "prerequisite_stage_evidence_sha256"
            ],
            attempt_id=context["attempt_id"],
            candidate_sha256=context["candidate_sha256"],
            candidate_design_sha256=context["candidate_design_sha256"],
            registry_entry_sha256=context["registry_entry_sha256"],
            request_sha256=context["request_sha256"],
            requested_stage=context["stage"],
            stage_access_manifest_sha256=context[
                "stage_access_manifest_sha256"
            ],
            registry_sha256=context["registry_sha256"],
            registry_tip_sha256=context["registry_tip_sha256"],
            semantic_checks=semantic_checks,
            semantic_receipt=semantic_receipt,
        )

    def to_dict(self) -> dict[str, Any]:
        validator_id = _safe_id(self.validator_id, "semantic validator_id")
        source_hash = _sha256(
            self.validator_source_sha256, "semantic validator_source_sha256"
        )
        prerequisite_stage = self.prerequisite_stage
        if prerequisite_stage not in {"development", "intermediate"}:
            raise SecFilingGemmaRevealStoreError(
                "Semantic prerequisite stage is invalid"
            )
        evidence_hash = _sha256(
            self.prerequisite_stage_evidence_sha256,
            "semantic prerequisite_stage_evidence_sha256",
        )
        attempt_id = _safe_id(self.attempt_id, "semantic attempt_id")
        candidate_hash = _sha256(self.candidate_sha256, "semantic candidate_sha256")
        candidate_design_hash = _sha256(
            self.candidate_design_sha256,
            "semantic candidate_design_sha256",
        )
        entry_hash = _sha256(
            self.registry_entry_sha256, "semantic registry_entry_sha256"
        )
        request_hash = _sha256(self.request_sha256, "semantic request_sha256")
        if self.requested_stage not in {"intermediate", "final"}:
            raise SecFilingGemmaRevealStoreError(
                "Semantic requested stage is invalid"
            )
        access_hash = _sha256(
            self.stage_access_manifest_sha256,
            "semantic stage_access_manifest_sha256",
        )
        registry_hash = _sha256(
            self.registry_sha256, "semantic registry_sha256"
        )
        registry_tip = _sha256(
            self.registry_tip_sha256, "semantic registry_tip_sha256"
        )
        semantic_checks = _semantic_checks(
            self.semantic_checks, "Semantic checks"
        )
        receipt = _json_value_copy(
            dict(_expect_mapping(self.semantic_receipt, "semantic receipt")),
            "semantic receipt",
        )
        if not receipt:
            raise SecFilingGemmaRevealStoreError("Semantic receipt cannot be empty")
        body = {
            "schema_version": SEMANTIC_PREREQUISITE_SCHEMA_VERSION,
            "validation_kind": "independent_semantic_prerequisite_replay",
            "validator_id": validator_id,
            "validator_source_sha256": source_hash,
            "prerequisite_stage": prerequisite_stage,
            "prerequisite_stage_evidence_sha256": evidence_hash,
            "attempt_id": attempt_id,
            "candidate_sha256": candidate_hash,
            "candidate_design_sha256": candidate_design_hash,
            "registry_entry_sha256": entry_hash,
            "request_sha256": request_hash,
            "requested_stage": self.requested_stage,
            "stage_access_manifest_sha256": access_hash,
            "registry_sha256": registry_hash,
            "registry_tip_sha256": registry_tip,
            "semantic_checks": list(semantic_checks),
            "semantic_receipt": receipt,
            "semantic_receipt_sha256": canonical_sha256(receipt),
            "semantic_validation_completed": True,
            "authorizes_outcome_access": False,
        }
        return {**body, "result_sha256": canonical_sha256(body)}


def _validate_semantic_result(
    result: Any,
    *,
    expected_context: Mapping[str, Any],
    expected_validator_id: str,
    expected_validator_source_sha256: str,
) -> dict[str, Any]:
    if type(result) is not SemanticPrerequisiteValidation:
        raise SecFilingGemmaRevealStoreError(
            "Prerequisite verifier must return SemanticPrerequisiteValidation, "
            "not an arbitrary truthy result"
        )
    value = result.to_dict()
    if (
        value["validator_id"] != _safe_id(
            expected_validator_id, "expected semantic validator_id"
        )
        or not _same_digest(
            value["validator_source_sha256"],
            _sha256(
                expected_validator_source_sha256,
                "expected semantic validator_source_sha256",
            ),
        )
    ):
        raise SecFilingGemmaRevealStoreError(
            "Prerequisite result is not from the independently pinned validator"
        )
    for key in (
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_sha256",
        "stage_access_manifest_sha256",
        "registry_sha256",
        "registry_tip_sha256",
    ):
        if value[key] != expected_context[key]:
            raise SecFilingGemmaRevealStoreError(
                f"Semantic prerequisite result is not bound to request field {key}"
            )
    semantic_receipt = _expect_mapping(
        value["semantic_receipt"],
        "semantic prerequisite audit receipt",
    )
    for receipt_key, context_key in (
        (
            "trusted_stage_content_pin_sha256",
            "trusted_stage_content_pin_sha256",
        ),
        (
            "trusted_stage_content_authentication_receipt_sha256",
            "trusted_stage_content_authentication_receipt_sha256",
        ),
        (
            "parent_consumption_binding_sha256",
            "parent_consumption_binding_sha256",
        ),
        (
            "authenticated_store_context_sha256",
            "authenticated_store_context_sha256",
        ),
    ):
        if semantic_receipt.get(receipt_key) != expected_context.get(context_key):
            raise SecFilingGemmaRevealStoreError(
                f"Semantic prerequisite audit lost store binding {receipt_key}"
            )
    if value["requested_stage"] != expected_context["stage"]:
        raise SecFilingGemmaRevealStoreError(
            "Semantic prerequisite result is not bound to the requested stage"
        )
    _semantic_checks(value["semantic_checks"], "Semantic prerequisite checks")
    body = {key: value[key] for key in value if key != "result_sha256"}
    if canonical_sha256(body) != value["result_sha256"]:
        raise SecFilingGemmaRevealStoreError(
            "Semantic prerequisite result hash is inconsistent"
        )
    return value


def _expected_context_for_consumed_request(
    request: Mapping[str, Any],
    *,
    trusted_stage_content_pin_sha256: str,
    trusted_stage_content_authentication_receipt_sha256: str,
    parent_consumption_binding_sha256: str | None,
    authenticated_store_context_sha256: str,
) -> dict[str, Any]:
    return {
        "prerequisite_stage": request["prerequisite_stage"],
        "prerequisite_stage_evidence_sha256": request[
            "prerequisite_stage_evidence_sha256"
        ],
        "attempt_id": request["attempt_id"],
        "candidate_sha256": request["candidate_sha256"],
        "candidate_design_sha256": request["candidate_design_sha256"],
        "registry_entry_sha256": request["registry_entry_sha256"],
        "request_sha256": request["request_sha256"],
        "stage": request["stage"],
        "stage_access_manifest_sha256": request[
            "stage_access_manifest_sha256"
        ],
        "registry_sha256": request["registry_sha256"],
        "registry_tip_sha256": request["registry_tip_sha256"],
        "trusted_stage_content_pin_sha256": trusted_stage_content_pin_sha256,
        "trusted_stage_content_authentication_receipt_sha256": (
            trusted_stage_content_authentication_receipt_sha256
        ),
        "parent_consumption_binding_sha256": parent_consumption_binding_sha256,
        "authenticated_store_context_sha256": authenticated_store_context_sha256,
    }


def _locate_parent_intermediate_predecessor(
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    child_request: Mapping[str, Any],
    prerequisite_stage_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = _expect_mapping(
        authenticated_store_snapshot,
        "parent predecessor store snapshot",
    )
    current_tip = _expect_mapping(
        independent_current_tip_anchor,
        "parent predecessor current tip",
    )
    if child_request["stage"] != "final" or child_request[
        "prerequisite_stage"
    ] != "intermediate":
        raise SecFilingGemmaRevealStoreError(
            "Only a final request may require an intermediate predecessor"
        )
    matching = [
        entry
        for entry in state["consumption_ledger"]["entries"]
        if entry["stage"] == "intermediate"
        and entry["attempt_id"] == child_request["attempt_id"]
        and entry["candidate_sha256"] == child_request["candidate_sha256"]
        and entry["registry_entry_sha256"]
        == child_request["registry_entry_sha256"]
        and entry["request"]["candidate_design_sha256"]
        == child_request["candidate_design_sha256"]
        and entry["request"]["registry_sha256"]
        == child_request["registry_sha256"]
        and entry["request"]["registry_tip_sha256"]
        == child_request["registry_tip_sha256"]
    ]
    if len(matching) != 1:
        raise SecFilingGemmaRevealStoreError(
            "Final request requires exactly one intermediate predecessor"
        )
    entry = matching[0]
    ledger = state["consumption_ledger"]
    if (
        entry != ledger["entries"][-1]
        or entry["entry_sha256"] != ledger["chain"]["tip_sha256"]
    ):
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor is not the exact consumption-ledger tip"
        )
    bundle = current_tip["authorization_bundles"].get(entry["request_sha256"])
    if bundle is None:
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor lacks its persisted authorization bundle"
        )
    output_receipt = current_tip["consumed_stage_output_receipts"].get(
        entry["request_sha256"]
    )
    if output_receipt is None:
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor lacks its persisted first-output receipt"
        )
    binding = _stage_evidence_output_binding(
        prerequisite_stage_evidence,
        authorization_grant=bundle["authorization_grant"],
    )
    if binding["output_stage_evidence_sha256"] != child_request[
        "prerequisite_stage_evidence_sha256"
    ]:
        raise SecFilingGemmaRevealStoreError(
            "Final request evidence differs from the parent output receipt"
        )
    try:
        validate_consumed_stage_output_receipt(
            output_receipt,
            authenticated_store_snapshot=state,
            independent_current_tip_anchor=current_tip,
            authorization_bundle=bundle,
            **binding,
        )
    except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor output receipt is not exact at the current tip"
        ) from exc
    return entry, bundle, output_receipt


def _build_parent_consumption_binding(
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    child_request: Mapping[str, Any],
    parent_entry: Mapping[str, Any],
    parent_bundle: Mapping[str, Any],
    parent_output_receipt: Mapping[str, Any],
    prerequisite_stage_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    state = _expect_mapping(
        authenticated_store_snapshot,
        "parent binding store snapshot",
    )
    current_tip = _expect_mapping(
        independent_current_tip_anchor,
        "parent binding current tip",
    )
    parent = _expect_mapping(parent_entry, "parent consumption entry")
    bundle = _expect_mapping(parent_bundle, "parent authorization bundle")
    output_receipt = _expect_mapping(
        parent_output_receipt,
        "parent consumed-stage output receipt",
    )
    parent_request = _expect_mapping(parent["request"], "parent reveal request")
    persisted_bundle = current_tip["authorization_bundles"].get(
        parent_request["request_sha256"]
    )
    if persisted_bundle != bundle:
        raise SecFilingGemmaRevealStoreError(
            "Parent authorization bundle is not the exact current-tip bundle"
        )
    if current_tip["consumed_stage_output_receipts"].get(
        parent_request["request_sha256"]
    ) != output_receipt:
        raise SecFilingGemmaRevealStoreError(
            "Parent output receipt is not the exact current-tip receipt"
        )
    parent_access = _expect_mapping(
        parent["stage_access_manifest"],
        "parent stage-access manifest",
    )
    parent_validation = _expect_mapping(
        parent["prerequisite_validation"],
        "parent prerequisite validation",
    )
    parent_audit = _expect_mapping(
        parent_validation["semantic_receipt"],
        "parent stage audit receipt",
    )
    parent_pin = current_tip["trusted_stage_content_pins"].get(
        parent_request["request_sha256"]
    )
    if parent_pin is None:
        raise SecFilingGemmaRevealStoreError(
            "Final request predecessor lacks its persisted trusted content pin"
        )
    if (
        parent_audit.get("schema_version") != STAGE_AUDIT_RECEIPT_SCHEMA_VERSION
        or parent_audit.get("prerequisite_stage") != "development"
        or parent_audit.get("requested_stage") != "intermediate"
        or parent_audit.get("stage_evidence_sha256")
        != parent_request["prerequisite_stage_evidence_sha256"]
        or parent_audit.get("candidate_sha256")
        != parent_request["candidate_sha256"]
        or parent_audit.get("trusted_stage_content_pin_sha256")
        != parent_pin.get("pin_sha256")
        or parent_audit.get("parent_consumption_binding_sha256") is not None
        or parent_audit.get("authorizes_outcome_access") is not False
    ):
        raise SecFilingGemmaRevealStoreError(
            "Parent audit receipt lost its exact stage, evidence, candidate, or pin binding"
        )
    audit_hash = _sha256(
        parent_audit.get("audit_receipt_sha256"),
        "parent audit receipt hash",
    )
    audit_body = {
        key: parent_audit[key]
        for key in parent_audit
        if key != "audit_receipt_sha256"
    }
    if not _same_digest(audit_hash, canonical_sha256(audit_body)):
        raise SecFilingGemmaRevealStoreError(
            "Parent audit receipt self-hash is inconsistent"
        )
    parent_authentication = validate_trusted_stage_content_authentication_receipt(
        parent_audit.get("trusted_stage_content_authentication")
    )
    if (
        parent_audit.get(
            "trusted_stage_content_authentication_receipt_sha256"
        )
        != parent_authentication["authentication_receipt_sha256"]
        or parent_validation["semantic_receipt_sha256"]
        != canonical_sha256(parent_audit)
    ):
        raise SecFilingGemmaRevealStoreError(
            "Parent audit receipt lost its authentication or semantic-receipt binding"
        )
    parent_store_context = _authenticated_store_verifier_context(
        trusted_stage_content_pin=parent_pin,
        trusted_stage_content_authentication=parent_authentication,
        parent_consumption_binding=None,
    )
    if (
        parent_store_context["authenticated_store_context_sha256"]
        != parent_audit.get("authenticated_store_context_sha256")
    ):
        raise SecFilingGemmaRevealStoreError(
            "Parent audit receipt does not bind its reconstructed store context"
        )
    parent_expected_context = _expected_context_for_consumed_request(
        parent_request,
        trusted_stage_content_pin_sha256=parent_pin["pin_sha256"],
        trusted_stage_content_authentication_receipt_sha256=parent_authentication[
            "authentication_receipt_sha256"
        ],
        parent_consumption_binding_sha256=None,
        authenticated_store_context_sha256=parent_store_context[
            "authenticated_store_context_sha256"
        ],
    )
    try:
        validate_consumed_stage_authorization_grant(
            bundle["authorization_grant"],
            authenticated_store_snapshot=state,
            external_store_state_pin=bundle["store_state_pin"],
            independent_current_tip_anchor=current_tip,
            expected_consumption_entry_sha256=parent["entry_sha256"],
            expected_request_sha256=parent_request["request_sha256"],
            expected_candidate_sha256=parent_request["candidate_sha256"],
            expected_stage="intermediate",
            expected_prerequisite_stage_evidence_sha256=parent_request[
                "prerequisite_stage_evidence_sha256"
            ],
            expected_stage_access_manifest_sha256=parent_request[
                "stage_access_manifest_sha256"
            ],
            expected_output_namespace=parent_access["output"]["namespace"],
        )
        output_binding = _stage_evidence_output_binding(
            prerequisite_stage_evidence,
            authorization_grant=bundle["authorization_grant"],
        )
        if output_binding["output_stage_evidence_sha256"] != child_request[
            "prerequisite_stage_evidence_sha256"
        ]:
            raise SecFilingGemmaStageAuthorizationError(
                "Parent output receipt differs from the child prerequisite evidence"
            )
        validate_consumed_stage_output_receipt(
            output_receipt,
            authenticated_store_snapshot=state,
            independent_current_tip_anchor=current_tip,
            authorization_bundle=bundle,
            **output_binding,
        )
    except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
        raise SecFilingGemmaRevealStoreError(
            "Parent authorization bundle is not valid at the current tip"
        ) from exc
    for request_field in (
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
    ):
        if parent_request[request_field] != child_request[request_field]:
            raise SecFilingGemmaRevealStoreError(
                f"Parent predecessor crossed child identity {request_field}"
            )
    child_section = {
        field: child_request[field]
        for field in (
            "request_sha256",
            "stage",
            "prerequisite_stage",
            "prerequisite_stage_evidence_sha256",
            "stage_access_manifest_sha256",
            "attempt_id",
            "candidate_sha256",
            "candidate_design_sha256",
            "registry_entry_sha256",
            "registry_sha256",
            "registry_tip_sha256",
        )
    }
    parent_section = {
        "entry_sha256": parent["entry_sha256"],
        "sequence": parent["sequence"],
        "request_sha256": parent_request["request_sha256"],
        "stage": parent_request["stage"],
        "prerequisite_stage": parent_request["prerequisite_stage"],
        "prerequisite_stage_evidence_sha256": parent_request[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": parent_request[
            "stage_access_manifest_sha256"
        ],
        "expected_context_sha256": canonical_sha256(parent_expected_context),
        "attempt_id": parent_request["attempt_id"],
        "candidate_sha256": parent_request["candidate_sha256"],
        "candidate_design_sha256": parent_request["candidate_design_sha256"],
        "registry_entry_sha256": parent_request["registry_entry_sha256"],
        "registry_sha256": parent_request["registry_sha256"],
        "registry_tip_sha256": parent_request["registry_tip_sha256"],
        "prerequisite_validation_result_sha256": parent_validation[
            "result_sha256"
        ],
        "semantic_receipt_sha256": parent_validation[
            "semantic_receipt_sha256"
        ],
        "audit_receipt": parent_audit,
        "audit_receipt_sha256": audit_hash,
        "trusted_stage_content_pin": parent_pin,
        "trusted_stage_content_pin_sha256": parent_pin["pin_sha256"],
        "authorization_bundle_sha256": bundle["bundle_sha256"],
        "authorization_grant_sha256": bundle["authorization_grant"][
            "authorization_grant_sha256"
        ],
        "store_pin_sha256": bundle["store_state_pin"]["store_pin_sha256"],
        "consumed_stage_output_receipt": output_receipt,
        "consumed_stage_output_receipt_sha256": output_receipt[
            "output_receipt_sha256"
        ],
    }
    authenticated_tip = {
        "store_state_sha256": state["state_sha256"],
        "state_snapshot_bytes_sha256": current_tip[
            "state_snapshot_bytes_sha256"
        ],
        "state_snapshot_byte_count": current_tip["state_snapshot_byte_count"],
        "consumption_ledger_sha256": state["consumption_ledger"][
            "ledger_sha256"
        ],
        "consumption_ledger_tip_sha256": state["consumption_ledger"]["chain"][
            "tip_sha256"
        ],
        "consumed_request_count": state["consumption_ledger"]["chain"][
            "consumed_request_count"
        ],
        "current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "current_tip_revision": current_tip["revision"],
        "consumed_stage_output_receipts": current_tip[
            "consumed_stage_output_receipts"
        ],
        "consumed_stage_output_receipts_sha256": canonical_sha256(
            current_tip["consumed_stage_output_receipts"]
        ),
    }
    body = {
        "schema_version": PARENT_CONSUMPTION_BINDING_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "binding_kind": "exact_prior_intermediate_consumption_and_grant",
        "child_request": child_section,
        "parent_consumption": parent_section,
        "authenticated_preconsumption_tip": authenticated_tip,
    }
    return {
        **body,
        "parent_consumption_binding_sha256": canonical_sha256(body),
    }


def _stage_evidence_sha256(evidence: Mapping[str, Any]) -> str:
    value = _json_value_copy(
        dict(_expect_mapping(evidence, "prerequisite stage evidence")),
        "prerequisite stage evidence",
    )
    if "stage_evidence_sha256" not in value:
        return canonical_sha256(value)
    observed = _sha256(
        value["stage_evidence_sha256"], "stage evidence self-hash"
    )
    body = {key: value[key] for key in value if key != "stage_evidence_sha256"}
    expected = canonical_sha256(body)
    if not _same_digest(observed, expected):
        raise SecFilingGemmaRevealStoreError(
            "Prerequisite stage evidence self-hash is inconsistent"
        )
    return expected


def _stage_evidence_output_binding(
    evidence: Mapping[str, Any],
    *,
    authorization_grant: Mapping[str, Any],
) -> dict[str, Any]:
    value = _json_value_copy(
        dict(_expect_mapping(evidence, "consumed-stage output evidence")),
        "consumed-stage output evidence",
    )
    grant = _expect_mapping(
        authorization_grant,
        "consumed-stage output authorization grant",
    )
    schema_version = _safe_id(
        value.get("schema_version"),
        "consumed-stage output evidence schema version",
    )
    prerequisite_stage = value.get("prerequisite_stage", value.get("stage"))
    prerequisite_stage = _safe_id(
        prerequisite_stage,
        "consumed-stage output evidence prerequisite stage",
    )
    candidate_manifest = value.get("candidate_manifest")
    candidate_sha256 = (
        candidate_manifest.get("candidate_sha256")
        if type(candidate_manifest) is dict
        else value.get("candidate_sha256")
    )
    candidate_hash = _sha256(
        candidate_sha256,
        "consumed-stage output evidence candidate hash",
    )
    parent_hash = value.get("parent_stage_evidence_sha256")
    if parent_hash is None:
        # Compact store tests and pre-run diagnostics may use a reduced
        # envelope.  The receipt still binds the only parent evidence hash
        # authorized by the exact grant; the production v3 evidence schema
        # carries this field explicitly and the authoritative verifier checks it.
        parent_hash = grant.get("prerequisite_stage_evidence_sha256")
    parent_hash = _sha256(
        parent_hash,
        "consumed-stage output evidence parent hash",
    )
    evidence_hash = _stage_evidence_sha256(value)
    try:
        canonical_bytes = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - detached above
        raise SecFilingGemmaRevealStoreError(
            "Consumed-stage output evidence is not canonical finite JSON"
        ) from exc
    if (
        prerequisite_stage != grant.get("stage")
        or parent_hash != grant.get("prerequisite_stage_evidence_sha256")
        or candidate_hash != grant.get("candidate_sha256")
    ):
        raise SecFilingGemmaRevealStoreError(
            "Consumed-stage output evidence crossed its authorization grant"
        )
    return {
        "output_stage_evidence_schema_version": schema_version,
        "output_stage_evidence_sha256": evidence_hash,
        "output_stage_evidence_document_sha256": hashlib.sha256(
            canonical_bytes
        ).hexdigest(),
        "output_stage_evidence_canonical_byte_count": len(canonical_bytes),
        "output_stage_evidence_prerequisite_stage": prerequisite_stage,
        "output_parent_stage_evidence_sha256": parent_hash,
        "output_candidate_sha256": candidate_hash,
    }


def _stage_access_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    value = _json_value_copy(
        dict(_expect_mapping(manifest, "stage access manifest")),
        "stage access manifest",
    )
    if "stage_access_manifest_sha256" not in value:
        raise SecFilingGemmaRevealStoreError(
            "Stage access manifest must carry its canonical self-hash"
        )
    observed = _sha256(
        value["stage_access_manifest_sha256"],
        "stage access manifest self-hash",
    )
    body = {
        key: value[key]
        for key in value
        if key != "stage_access_manifest_sha256"
    }
    expected = canonical_sha256(body)
    if not _same_digest(observed, expected):
        raise SecFilingGemmaRevealStoreError(
            "Stage access manifest self-hash is inconsistent"
        )
    return expected


def _validate_consumption_ledger(
    ledger: Mapping[str, Any], *, anchor: Mapping[str, Any]
) -> dict[str, Any]:
    value = _expect_mapping(ledger, "consumption ledger")
    _expect_keys(
        value,
        {"schema_version", "entries", "chain", "ledger_sha256"},
        "consumption ledger",
    )
    if value["schema_version"] != CONSUMPTION_LEDGER_SCHEMA_VERSION:
        raise SecFilingGemmaRevealStoreError("Unknown consumption-ledger schema")
    if not isinstance(value["entries"], list):
        raise SecFilingGemmaRevealStoreError("Consumption entries must be a list")

    entries: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    seen_stages: set[tuple[str, str]] = set()
    current_tip = _consumption_genesis(anchor)
    final_count = 0
    intermediate_candidates: set[tuple[str, str, str]] = set()
    entry_keys = {
        "schema_version",
        "sequence",
        "request_sha256",
        "request",
        "stage_access_manifest",
        "stage",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "prerequisite_validation",
        "prior_tip_sha256",
        "final_touch_delta",
        "cumulative_actual_final_touch_count",
        "entry_sha256",
    }
    validation_keys = {
        "schema_version",
        "validation_kind",
        "validator_id",
        "validator_source_sha256",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_sha256",
        "requested_stage",
        "stage_access_manifest_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "semantic_checks",
        "semantic_receipt",
        "semantic_receipt_sha256",
        "semantic_validation_completed",
        "authorizes_outcome_access",
        "result_sha256",
    }

    for sequence, raw_entry in enumerate(value["entries"], start=1):
        entry = _expect_mapping(raw_entry, f"consumption entry {sequence}")
        _expect_keys(entry, entry_keys, f"consumption entry {sequence}")
        if (
            entry["schema_version"] != CONSUMPTION_ENTRY_SCHEMA_VERSION
            or _strict_int(entry["sequence"], "consumption sequence", minimum=1)
            != sequence
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumption entry sequence or schema is invalid"
            )
        request = _expect_mapping(entry["request"], "consumed reveal request")
        request_hash = _request_body_hash(request)
        if entry["request_sha256"] != request_hash:
            raise SecFilingGemmaRevealStoreError(
                "Consumption entry does not bind its exact request"
            )
        access_manifest = _expect_mapping(
            entry["stage_access_manifest"], "consumed stage access manifest"
        )
        if (
            _stage_access_manifest_sha256(access_manifest)
            != request["stage_access_manifest_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumption entry does not bind its exact stage access manifest"
            )
        stage = entry["stage"]
        attempt_id = entry["attempt_id"]
        candidate_hash = entry["candidate_sha256"]
        registry_entry_hash = entry["registry_entry_sha256"]
        if (
            stage != request["stage"]
            or attempt_id != request["attempt_id"]
            or candidate_hash != request["candidate_sha256"]
            or registry_entry_hash != request["registry_entry_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumption identity is not bound to the stored request"
            )
        _safe_id(attempt_id, "consumption attempt_id")
        _sha256(candidate_hash, "consumption candidate_sha256")
        _sha256(registry_entry_hash, "consumption registry_entry_sha256")
        if request_hash in seen_requests or (attempt_id, stage) in seen_stages:
            raise SecFilingGemmaRevealStoreError(
                "A reveal request or candidate stage was consumed more than once"
            )

        validation = _expect_mapping(
            entry["prerequisite_validation"], "stored prerequisite validation"
        )
        _expect_keys(validation, validation_keys, "stored prerequisite validation")
        validation_body = {
            key: validation[key] for key in validation if key != "result_sha256"
        }
        semantic_receipt = _expect_mapping(
            validation["semantic_receipt"], "stored semantic receipt"
        )
        if (
            validation["schema_version"] != SEMANTIC_PREREQUISITE_SCHEMA_VERSION
            or validation["validation_kind"]
            != "independent_semantic_prerequisite_replay"
            or validation["semantic_validation_completed"] is not True
            or validation["authorizes_outcome_access"] is not False
            or canonical_sha256(validation_body) != validation["result_sha256"]
            or not semantic_receipt
            or canonical_sha256(semantic_receipt)
            != validation["semantic_receipt_sha256"]
            or _semantic_checks(
                validation["semantic_checks"], "Stored semantic checks"
            )
            != REQUIRED_SEMANTIC_CHECKS
        ):
            raise SecFilingGemmaRevealStoreError(
                "Stored prerequisite validation is not canonical semantic evidence"
            )
        _safe_id(validation["validator_id"], "stored semantic validator_id")
        _sha256(
            validation["validator_source_sha256"],
            "stored semantic validator_source_sha256",
        )
        _sha256(
            validation["candidate_design_sha256"],
            "stored semantic candidate_design_sha256",
        )
        _sha256(validation["request_sha256"], "stored semantic request_sha256")
        _sha256(
            validation["stage_access_manifest_sha256"],
            "stored semantic stage_access_manifest_sha256",
        )
        _sha256(
            validation["registry_sha256"], "stored semantic registry_sha256"
        )
        _sha256(
            validation["registry_tip_sha256"],
            "stored semantic registry_tip_sha256",
        )
        for request_key, validation_key in (
            ("prerequisite_stage", "prerequisite_stage"),
            (
                "prerequisite_stage_evidence_sha256",
                "prerequisite_stage_evidence_sha256",
            ),
            ("attempt_id", "attempt_id"),
            ("candidate_sha256", "candidate_sha256"),
            ("candidate_design_sha256", "candidate_design_sha256"),
            ("registry_entry_sha256", "registry_entry_sha256"),
            ("request_sha256", "request_sha256"),
            ("stage_access_manifest_sha256", "stage_access_manifest_sha256"),
            ("registry_sha256", "registry_sha256"),
            ("registry_tip_sha256", "registry_tip_sha256"),
        ):
            if request[request_key] != validation[validation_key]:
                raise SecFilingGemmaRevealStoreError(
                    "Stored semantic validation lost its request binding"
                )
        if validation["requested_stage"] != request["stage"]:
            raise SecFilingGemmaRevealStoreError(
                "Stored semantic validation lost its requested-stage binding"
            )

        prior_tip = _sha256(entry["prior_tip_sha256"], "consumption prior tip")
        if not _same_digest(prior_tip, current_tip):
            raise SecFilingGemmaRevealStoreError(
                "Consumed-request hash chain is broken or rolled back"
            )
        delta = _strict_int(entry["final_touch_delta"], "final touch delta")
        if delta != (1 if stage == "final" else 0):
            raise SecFilingGemmaRevealStoreError(
                "Only actual final-request consumption may increment final touches"
            )
        candidate_key = (attempt_id, candidate_hash, registry_entry_hash)
        if stage == "final" and candidate_key not in intermediate_candidates:
            raise SecFilingGemmaRevealStoreError(
                "Final request cannot precede intermediate request consumption"
            )
        final_count += delta
        if entry["cumulative_actual_final_touch_count"] != final_count:
            raise SecFilingGemmaRevealStoreError(
                "Cumulative actual final-touch count is inconsistent"
            )
        body = {key: entry[key] for key in entry if key != "entry_sha256"}
        expected_entry_hash = canonical_sha256(body)
        if entry["entry_sha256"] != expected_entry_hash:
            raise SecFilingGemmaRevealStoreError(
                "Consumed-request entry hash is inconsistent"
            )

        normalized = _json_value_copy(dict(entry), "consumption entry")
        entries.append(normalized)
        seen_requests.add(request_hash)
        seen_stages.add((attempt_id, stage))
        if stage == "intermediate":
            intermediate_candidates.add(candidate_key)
        current_tip = expected_entry_hash

    expected = _consumption_ledger(entries, tip_sha256=current_tip, anchor=anchor)
    if value != expected:
        raise SecFilingGemmaRevealStoreError(
            "Consumption ledger count, tip, or final-touch total is inconsistent"
        )
    return expected


def _validate_state(
    state: Mapping[str, Any], *, expected_anchor: Mapping[str, Any]
) -> dict[str, Any]:
    value = _expect_mapping(state, "reveal-store state")
    _expect_keys(
        value,
        {
            "schema_version",
            "contract_version",
            "anchor",
            "latest_registry",
            "latest_registry_pin",
            "consumption_ledger",
            "state_sha256",
        },
        "reveal-store state",
    )
    if (
        value["schema_version"] != STORE_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store state belongs to another schema or contract"
        )
    anchor = _expect_mapping(value["anchor"], "stored anchor")
    if anchor != expected_anchor:
        raise SecFilingGemmaRevealStoreError(
            "Reveal-store state does not bind the tracked genesis anchors"
        )
    registry = _expect_mapping(value["latest_registry"], "latest registry")
    pin = _expect_mapping(value["latest_registry_pin"], "latest registry pin")
    try:
        validate_reveal_registry(registry, external_pin=pin)
    except SecFilingGemmaRevealRegistryError as exc:
        raise SecFilingGemmaRevealStoreError(
            "Authoritative registry and pin are stale, forked, rolled back, or invalid"
        ) from exc
    ledger = _validate_consumption_ledger(
        _expect_mapping(value["consumption_ledger"], "consumption ledger"),
        anchor=anchor,
    )
    latest_entries = registry["entries"]
    for consumed in ledger["entries"]:
        request = consumed["request"]
        registered_count = _strict_int(
            request["registered_entry_count"],
            "consumed request registered_entry_count",
            minimum=1,
        )
        if registered_count > len(latest_entries):
            raise SecFilingGemmaRevealStoreError(
                "Latest registry was rolled back behind a consumed request"
            )
        registered_entry = latest_entries[registered_count - 1]
        if (
            registered_entry["entry_sha256"]
            != request["registry_entry_sha256"]
            or registered_entry["candidate_sha256"]
            != request["candidate_sha256"]
            or registered_entry["attempt_id"] != request["attempt_id"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumed request is not an ancestor of the latest registry"
            )
        prefix = copy.deepcopy(dict(registry))
        prefix["entries"] = copy.deepcopy(latest_entries[:registered_count])
        prefix["chain"]["tip_sha256"] = registered_entry["entry_sha256"]
        prefix["chain"]["registered_entry_count"] = registered_count
        prefix_body = {
            key: prefix[key] for key in prefix if key != "registry_sha256"
        }
        prefix["registry_sha256"] = canonical_sha256(prefix_body)
        try:
            prefix_pin = derive_registry_pin(prefix)
        except SecFilingGemmaRevealRegistryError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Consumed request registry ancestry is not canonical"
            ) from exc
        if (
            request["registry_sha256"] != prefix["registry_sha256"]
            or request["registry_tip_sha256"] != prefix_pin["tip_sha256"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Consumed request does not bind its exact registry ancestor"
            )
    expected = _state_snapshot(
        anchor=anchor,
        latest_registry=registry,
        latest_pin=pin,
        consumption_ledger=ledger,
    )
    if value != expected:
        raise SecFilingGemmaRevealStoreError(
            "Authoritative reveal-store state hash is inconsistent"
        )
    return expected


class SecFilingGemmaRevealStore:
    """Atomic local store for registry appends and one-shot reveal requests."""

    def __init__(
        self,
        *,
        repository_root: Path,
        store_directory: Path,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        if isinstance(lock_timeout_seconds, bool) or not isinstance(
            lock_timeout_seconds, (int, float)
        ):
            raise TypeError("lock_timeout_seconds must be a positive number")
        if not (0.0 < float(lock_timeout_seconds) <= 60.0):
            raise SecFilingGemmaRevealStoreError(
                "lock_timeout_seconds must be in (0, 60]"
            )
        self._repository_root = _secure_directory(
            Path(repository_root), create=False, location="repository_root"
        )
        self._store_directory = Path(
            os.path.abspath(os.fspath(store_directory))
        )
        self._lock_timeout_seconds = float(lock_timeout_seconds)

    @property
    def repository_root(self) -> Path:
        return self._repository_root

    @property
    def store_directory(self) -> Path:
        return self._store_directory

    @property
    def lock_timeout_seconds(self) -> float:
        return self._lock_timeout_seconds

    @property
    def state_path(self) -> Path:
        return self.store_directory / STATE_FILENAME

    @property
    def current_tip_anchor_path(self) -> Path:
        return self.store_directory / CURRENT_TIP_ANCHOR_FILENAME

    @property
    def restore_pending_path(self) -> Path:
        return self.store_directory / RESTORE_PENDING_FILENAME

    @property
    def lock_path(self) -> Path:
        return self.store_directory / LOCK_FILENAME

    def _locked(self) -> _ExclusiveFileLock:
        secured = _secure_directory(
            self._store_directory,
            create=True,
            location="reveal-store directory",
        )
        if secured != self._store_directory:
            raise SecFilingGemmaRevealStoreError(
                "Reveal-store directory identity changed"
            )
        return _ExclusiveFileLock(
            self.lock_path, timeout_seconds=self._lock_timeout_seconds
        )

    def _recover_pending_restore_locked(
        self, tracked_anchor: Mapping[str, Any]
    ) -> None:
        path = self.restore_pending_path
        if not path.exists() and not path.is_symlink():
            return
        payload = _read_regular_bytes(
            path,
            "pending reveal-store restore transaction",
            max_bytes=MAX_RESTORE_PENDING_FILE_BYTES,
        )
        parsed = _strict_json_bytes(
            payload, "pending reveal-store restore transaction"
        )
        recovery = _validated_restore_pending_document(
            parsed, tracked_anchor=tracked_anchor
        )
        # The recovery file remains authoritative until both target files are
        # exact. A stop after either replace simply replays this idempotently.
        _atomic_replace(self.state_path, recovery["state_bytes"])
        _atomic_replace(
            self.current_tip_anchor_path, recovery["tip_anchor_bytes"]
        )
        if (
            _read_regular_bytes(
                self.state_path,
                "recovered reveal-store state",
                max_bytes=MAX_STATE_FILE_BYTES,
            )
            != recovery["state_bytes"]
            or _read_regular_bytes(
                self.current_tip_anchor_path,
                "recovered reveal-store current-tip anchor",
                max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
            )
            != recovery["tip_anchor_bytes"]
        ):
            raise SecFilingGemmaRevealStoreError(
                "Recovered reveal-store files differ from the restore transaction"
            )
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(
                metadata.st_mode
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Restore transaction path changed file type"
                )
            path.unlink()
        except OSError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Completed restore transaction could not be retired"
            ) from exc
        if path.exists() or path.is_symlink():
            raise SecFilingGemmaRevealStoreError(
                "Completed restore transaction remains present"
            )

    def _read_state_and_tip_locked(
        self,
        tracked_anchor: Mapping[str, Any],
        *,
        recover_pending_restore: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
        if recover_pending_restore:
            self._recover_pending_restore_locked(tracked_anchor)
        try:
            state_payload = _read_regular_bytes(
                self.state_path,
                "authoritative reveal-store state",
                max_bytes=MAX_STATE_FILE_BYTES,
            )
        except SecFilingGemmaRevealStoreError as exc:
            if not self.state_path.exists() and not self.state_path.is_symlink():
                raise SecFilingGemmaRevealStoreError(
                    "Reveal store is not initialized"
                ) from exc
            raise
        state_parsed = _strict_json_bytes(
            state_payload, "authoritative reveal-store state"
        )
        try:
            state_parsed = detach_untrusted_stage_json(
                state_parsed, "authoritative reveal-store state"
            )
        except Exception as exc:
            raise SecFilingGemmaRevealStoreError(
                "Authoritative reveal-store state exceeds fixed allocation bounds"
            ) from exc
        state = _validate_state(
            _expect_mapping(state_parsed, "authoritative reveal-store state"),
            expected_anchor=tracked_anchor,
        )
        try:
            tip_payload = _read_regular_bytes(
                self.current_tip_anchor_path,
                "independent reveal-store current-tip anchor",
                max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
            )
        except SecFilingGemmaRevealStoreError as exc:
            if (
                not self.current_tip_anchor_path.exists()
                and not self.current_tip_anchor_path.is_symlink()
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Independent reveal-store current-tip anchor is missing"
                ) from exc
            raise
        tip_parsed = _strict_json_bytes(
            tip_payload, "independent reveal-store current-tip anchor"
        )
        tip_mapping = _expect_mapping(
            tip_parsed, "independent reveal-store current-tip anchor"
        )
        if tip_mapping.get("schema_version") == CURRENT_TIP_PENDING_SCHEMA_VERSION:
            pending = _validated_pending_current_tip_document(tip_mapping)
            prior = pending["prior_tip_anchor"]
            next_anchor = pending["next_tip_anchor"]
            if _state_bytes_match_tip_anchor(state_payload, state, next_anchor):
                resolved = next_anchor
            elif prior is not None and _state_bytes_match_tip_anchor(
                state_payload, state, prior
            ):
                resolved = prior
            else:
                raise SecFilingGemmaRevealStoreError(
                    "Interrupted current-tip transaction matches neither its prior nor next state"
                )
            try:
                resolved = validate_reveal_store_current_tip_anchor(
                    state, resolved
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Interrupted current-tip transaction cannot authenticate its state"
                ) from exc
            _atomic_replace(
                self.current_tip_anchor_path,
                _encoded_state(resolved),
            )
            tip_payload = _read_regular_bytes(
                self.current_tip_anchor_path,
                "resolved reveal-store current-tip anchor",
                max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
            )
            if tip_payload != _encoded_state(resolved):
                raise SecFilingGemmaRevealStoreError(
                    "Resolved current-tip anchor bytes are inconsistent"
                )
            tip = resolved
        else:
            try:
                tip = validate_reveal_store_current_tip_anchor(
                    state, tip_mapping
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Independent current-tip anchor does not authenticate the latest state"
                ) from exc
        return state, tip, state_payload, tip_payload

    def _read_state_locked(self, anchor: Mapping[str, Any]) -> dict[str, Any]:
        state, _tip, _state_bytes, _tip_bytes = self._read_state_and_tip_locked(
            anchor
        )
        return state

    def _commit_state_and_tip_locked(
        self,
        *,
        tracked_anchor: Mapping[str, Any],
        prior_tip_anchor: Mapping[str, Any],
        next_state: Mapping[str, Any],
        authorization_bundle: Mapping[str, Any] | None = None,
        trusted_stage_content_pin: Mapping[str, Any] | None = None,
        consumed_stage_output_receipt: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        validated_state = _validate_state(
            _expect_mapping(next_state, "next reveal-store state"),
            expected_anchor=tracked_anchor,
        )
        bundles = copy.deepcopy(prior_tip_anchor["authorization_bundles"])
        pins = copy.deepcopy(prior_tip_anchor["trusted_stage_content_pins"])
        output_receipts = copy.deepcopy(
            prior_tip_anchor["consumed_stage_output_receipts"]
        )
        transition_payload_count = sum(
            value is not None
            for value in (
                authorization_bundle,
                trusted_stage_content_pin,
                consumed_stage_output_receipt,
            )
        )
        if transition_payload_count > 1:
            raise SecFilingGemmaRevealStoreError(
                "Trusted pin, authorization bundle, and output receipt require separate transitions"
            )
        if trusted_stage_content_pin is not None:
            pin = _exact_caller_dict(
                trusted_stage_content_pin,
                "trusted stage-content pin",
            )
            request_hash = _sha256(
                pin.get("request_sha256"),
                "trusted stage-content pin request hash",
            )
            if request_hash in pins and pins[request_hash] != pin:
                raise SecFilingGemmaRevealStoreError(
                    "A persisted trusted stage-content pin cannot be replaced"
                )
            pins[request_hash] = pin
        if authorization_bundle is not None:
            bundle = _exact_caller_dict(
                authorization_bundle, "authorization bundle"
            )
            request_hash = _sha256(
                bundle.get("authorization_grant", {}).get("request_sha256"),
                "authorization bundle request hash",
            )
            if request_hash in bundles and bundles[request_hash] != bundle:
                raise SecFilingGemmaRevealStoreError(
                    "A persisted authorization bundle cannot be replaced"
                )
            bundles[request_hash] = bundle
        if consumed_stage_output_receipt is not None:
            output_receipt = _exact_caller_dict(
                consumed_stage_output_receipt,
                "consumed-stage output receipt",
            )
            request_hash = _sha256(
                output_receipt.get("request_sha256"),
                "consumed-stage output receipt request hash",
            )
            if (
                request_hash in output_receipts
                and output_receipts[request_hash] != output_receipt
            ):
                raise SecFilingGemmaRevealStoreError(
                    "A persisted consumed-stage output receipt cannot be replaced"
                )
            output_receipts[request_hash] = output_receipt
        try:
            next_tip = build_reveal_store_current_tip_anchor(
                validated_state,
                revision=prior_tip_anchor["revision"] + 1,
                previous_tip_anchor_sha256=prior_tip_anchor["tip_anchor_sha256"],
                authorization_bundles=bundles,
                trusted_stage_content_pins=pins,
                consumed_stage_output_receipts=output_receipts,
            )
        except SecFilingGemmaStageAuthorizationError as exc:
            raise SecFilingGemmaRevealStoreError(
                "Could not build the next independent current-tip anchor"
            ) from exc
        pending = _pending_current_tip_document(
            prior_tip_anchor=prior_tip_anchor,
            next_tip_anchor=next_tip,
        )
        state_bytes = _encoded_state(validated_state)
        pending_bytes = _encoded_state(pending)
        next_tip_bytes = _encoded_state(next_tip)
        if len(state_bytes) > MAX_STATE_FILE_BYTES:
            raise SecFilingGemmaRevealStoreError(
                "Next reveal-store state exceeds its safety limit"
            )
        if max(len(pending_bytes), len(next_tip_bytes)) > MAX_CURRENT_TIP_ANCHOR_FILE_BYTES:
            raise SecFilingGemmaRevealStoreError(
                "Next current-tip transaction exceeds its safety limit"
            )

        # The pending anchor is a write-ahead CAS record.  A crash before the
        # state replace resolves to ``prior``; a crash after it resolves to
        # ``next``.  Thus the consumed entry and exact persisted bundle become
        # visible as one logical transaction despite using two regular files.
        _atomic_replace(self.current_tip_anchor_path, pending_bytes)
        _atomic_replace(self.state_path, state_bytes)
        _atomic_replace(self.current_tip_anchor_path, next_tip_bytes)
        committed_state, committed_tip, _state_bytes, _tip_bytes = (
            self._read_state_and_tip_locked(tracked_anchor)
        )
        if committed_state != validated_state or committed_tip != next_tip:
            raise SecFilingGemmaRevealStoreError(
                "Committed state/current-tip transaction differs from its CAS target"
            )
        return committed_state, committed_tip

    def initialize(self) -> dict[str, Any]:
        """Create genesis state once, or validate and return existing state."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            self._recover_pending_restore_locked(anchor)
            state_exists = self.state_path.exists() or self.state_path.is_symlink()
            tip_exists = (
                self.current_tip_anchor_path.exists()
                or self.current_tip_anchor_path.is_symlink()
            )
            if state_exists and tip_exists:
                return self._read_state_locked(anchor)
            registry = build_initial_reveal_registry()
            pin = derive_registry_pin(registry)
            state = _state_snapshot(
                anchor=anchor,
                latest_registry=registry,
                latest_pin=pin,
                consumption_ledger=_initial_consumption_ledger(anchor),
            )
            try:
                tip = build_reveal_store_current_tip_anchor(
                    state,
                    revision=0,
                    previous_tip_anchor_sha256=None,
                    authorization_bundles={},
                    trusted_stage_content_pins={},
                    consumed_stage_output_receipts={},
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Could not build the genesis current-tip anchor"
                ) from exc
            pending = _pending_current_tip_document(
                prior_tip_anchor=None,
                next_tip_anchor=tip,
            )
            if state_exists and not tip_exists:
                raise SecFilingGemmaRevealStoreError(
                    "Reveal store has state without its genesis write-ahead anchor"
                )
            if tip_exists and not state_exists:
                pending_bytes = _read_regular_bytes(
                    self.current_tip_anchor_path,
                    "interrupted genesis current-tip transaction",
                    max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
                )
                parsed = _strict_json_bytes(
                    pending_bytes, "interrupted genesis current-tip transaction"
                )
                observed_pending = _validated_pending_current_tip_document(
                    parsed
                )
                if observed_pending != pending:
                    raise SecFilingGemmaRevealStoreError(
                        "Incomplete reveal-store pair is not the deterministic genesis transaction"
                    )
                _atomic_replace(self.state_path, _encoded_state(state))
                _atomic_replace(
                    self.current_tip_anchor_path, _encoded_state(tip)
                )
                return self._read_state_locked(anchor)
            _atomic_replace(self.current_tip_anchor_path, _encoded_state(pending))
            _atomic_replace(self.state_path, _encoded_state(state))
            _atomic_replace(self.current_tip_anchor_path, _encoded_state(tip))
            return self._read_state_locked(anchor)

    def load(self) -> dict[str, Any]:
        """Load and fully authenticate the latest state without mutating it."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            return self._read_state_locked(anchor)

    def load_current_tip_anchor(self) -> dict[str, Any]:
        """Load the independent latest-tip proof for downstream validation."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            _state, tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            return copy.deepcopy(tip)

    def compare_and_swap_append(
        self,
        *,
        transition: Mapping[str, Any],
        appended_registry: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically apply exactly one child registry transition.

        The transition's expected prior pin must equal the independently loaded
        authoritative pin.  A stale, forked, skipped, or rollback transition is
        rejected without changing the store.
        """

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(anchor)
            )
            # Authenticate and capture both files before touching caller data.
            if type(transition) is not dict or type(appended_registry) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Registry CAS inputs must be exact built-in dicts"
                )
            try:
                detached_cas = detach_untrusted_stage_json(
                    {
                        "transition": transition,
                        "appended_registry": appended_registry,
                    },
                    "registry CAS caller bundle",
                )
            except Exception as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Registry CAS inputs exceed fixed allocation bounds"
                ) from exc
            transition_value = detached_cas["transition"]
            appended_value = detached_cas["appended_registry"]
            try:
                validate_registry_pin_transition(
                    transition_value,
                    prior_registry=current["latest_registry"],
                    external_prior_pin=current["latest_registry_pin"],
                    appended_registry=appended_value,
                )
                next_pin = derive_registry_pin(appended_value)
            except SecFilingGemmaRevealRegistryError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Registry compare-and-swap rejected a stale, forked, or rollback transition"
                ) from exc
            next_state = _state_snapshot(
                anchor=anchor,
                latest_registry=appended_value,
                latest_pin=next_pin,
                consumption_ledger=current["consumption_ledger"],
            )
            committed, _next_tip = self._commit_state_and_tip_locked(
                tracked_anchor=anchor,
                prior_tip_anchor=current_tip,
                next_state=next_state,
            )
            return committed

    def consume_request(
        self,
        request: Mapping[str, Any],
        *,
        candidate_manifest: Mapping[str, Any],
        stage: str,
        stage_access_manifest: Mapping[str, Any],
        prerequisite_stage_evidence: Mapping[str, Any],
        _issue_authorization_grant: bool = False,
    ) -> dict[str, Any]:
        """Validate and atomically consume one non-authorizing stage request.

        The fixed candidate-bound verifier executes while the exclusive lock is
        held. It receives bounded detached copies of the prerequisite evidence,
        the actual self-hashed stage-access manifest, and the expected request
        context plus a reveal-store-authenticated trusted-content pin and, for
        final requests, the exact prior intermediate consumption/grant binding.
        It must return :class:`SemanticPrerequisiteValidation`.  Failure,
        replay, verifier mutation of the state file, or any binding mismatch
        leaves no consumption entry. A successfully precommitted trusted pin is
        deliberately retained across verifier failure and reused on an exact
        retry; it cannot authorize access by itself.

        There is deliberately no caller-supplied validator parameter. Until
        the fixed verifier can complete every frozen check, it raises and this
        method cannot consume a request.
        """
        with self._locked():
            locked_repository_root = self._repository_root
            locked_store_directory = self._store_directory
            locked_state_path = self.state_path
            locked_tip_anchor_path = self.current_tip_anchor_path
            locked_restore_pending_path = self.restore_pending_path
            locked_lock_path = self.lock_path
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            (
                current,
                current_tip,
                original_state_bytes,
                original_tip_anchor_bytes,
            ) = self._read_state_and_tip_locked(anchor)
            # Both independent files are authenticated and captured before any
            # authorization-critical caller object is examined.
            if type(_issue_authorization_grant) is not bool:
                raise SecFilingGemmaRevealStoreError(
                    "Internal grant-issuance flag must be an exact boolean"
                )
            caller_objects = {
                "request": request,
                "candidate_manifest": candidate_manifest,
                "stage_access_manifest": stage_access_manifest,
                "prerequisite_stage_evidence": prerequisite_stage_evidence,
            }
            if any(type(item) is not dict for item in caller_objects.values()):
                raise SecFilingGemmaRevealStoreError(
                    "Authorization-critical inputs must be exact built-in dicts"
                )
            # One shared no-copy budget rejects an already-oversized bundle
            # before allocation.  The bounded detacher then rechecks every
            # limit while copying, so concurrent mutation cannot insert an
            # unchecked deep or oversized value between check and copy.
            try:
                preflight_untrusted_stage_json(
                    caller_objects, "reveal request caller bundle"
                )
                detached_objects = detach_untrusted_stage_json(
                    caller_objects, "reveal request caller bundle"
                )
            except Exception as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request inputs exceed the fixed verifier allocation bounds"
                ) from exc
            if type(detached_objects) is not dict:  # pragma: no cover - fixed wrapper
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request inputs could not be detached"
                )
            request_value = detached_objects["request"]
            candidate_value = detached_objects["candidate_manifest"]
            if type(stage) is not str:
                raise SecFilingGemmaRevealStoreError(
                    "requested stage must be an exact built-in string"
                )
            access_manifest = detached_objects["stage_access_manifest"]
            access_hash = _stage_access_manifest_sha256(access_manifest)
            evidence = detached_objects["prerequisite_stage_evidence"]
            evidence_hash = _stage_evidence_sha256(evidence)
            try:
                request_hash = validate_single_candidate_reveal_request(
                    request_value,
                    registry=current["latest_registry"],
                    external_pin=current["latest_registry_pin"],
                    candidate_manifest=candidate_value,
                    stage=stage,
                    stage_access_manifest_sha256=access_hash,
                    prerequisite_stage_evidence_sha256=evidence_hash,
                )
            except SecFilingGemmaRevealRegistryError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request is altered, stale, non-authorizing, or not current-tip bound"
                ) from exc
            ledger = current["consumption_ledger"]
            matching_entries = [
                entry
                for entry in ledger["entries"]
                if (
                entry["request_sha256"] == request_hash
                or (
                    entry["attempt_id"] == request_value["attempt_id"]
                    and entry["stage"] == stage
                )
                )
            ]
            if matching_entries:
                if _issue_authorization_grant:
                    existing_entry = matching_entries[0]
                    existing_bundle = current_tip["authorization_bundles"].get(
                        request_hash
                    )
                    if (
                        existing_entry["request_sha256"] == request_hash
                        and existing_entry["entry_sha256"]
                        == ledger["chain"]["tip_sha256"]
                        and existing_bundle is not None
                        and existing_bundle["authenticated_store_snapshot"]
                        == current
                    ):
                        try:
                            validate_consumed_stage_authorization_grant(
                                existing_bundle["authorization_grant"],
                                authenticated_store_snapshot=current,
                                external_store_state_pin=existing_bundle[
                                    "store_state_pin"
                                ],
                                independent_current_tip_anchor=current_tip,
                                expected_consumption_entry_sha256=existing_entry[
                                    "entry_sha256"
                                ],
                                expected_request_sha256=request_hash,
                                expected_candidate_sha256=request_value[
                                    "candidate_sha256"
                                ],
                                expected_stage=stage,
                                expected_prerequisite_stage_evidence_sha256=(
                                    evidence_hash
                                ),
                                expected_stage_access_manifest_sha256=access_hash,
                                expected_output_namespace=access_manifest["output"][
                                    "namespace"
                                ],
                            )
                        except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                            raise SecFilingGemmaRevealStoreError(
                                "Persisted retry bundle failed current-tip authentication"
                            ) from exc
                        return copy.deepcopy(existing_bundle)
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request or candidate stage was already consumed"
                )
            parent_entry: dict[str, Any] | None = None
            parent_bundle: dict[str, Any] | None = None
            parent_output_receipt: dict[str, Any] | None = None
            if stage == "final":
                # Perform this exact predecessor/grant check before the monotonic
                # final-request pin precommit. A missing or ungranted parent must
                # leave no final-stage authorization material behind.
                (
                    parent_entry,
                    parent_bundle,
                    parent_output_receipt,
                ) = _locate_parent_intermediate_predecessor(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    child_request=request_value,
                    prerequisite_stage_evidence=evidence,
                )

            try:
                trusted_content_pin = derive_reveal_store_trusted_stage_content_pin(
                    current,
                    reveal_request=request_value,
                    stage_access_manifest=access_manifest,
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Reveal store could not derive the exact trusted stage-content pin"
                ) from exc
            persisted_pin = current_tip["trusted_stage_content_pins"].get(
                request_hash
            )
            if persisted_pin is None:
                try:
                    current, current_tip = self._commit_state_and_tip_locked(
                        tracked_anchor=anchor,
                        prior_tip_anchor=current_tip,
                        next_state=current,
                        trusted_stage_content_pin=trusted_content_pin,
                    )
                except SecFilingGemmaStageAuthorizationError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Trusted stage-content pin precommit failed closed"
                    ) from exc
                (
                    current,
                    current_tip,
                    original_state_bytes,
                    original_tip_anchor_bytes,
                ) = self._read_state_and_tip_locked(anchor)
                persisted_pin = current_tip["trusted_stage_content_pins"].get(
                    request_hash
                )
            if persisted_pin != trusted_content_pin:
                raise SecFilingGemmaRevealStoreError(
                    "Current tip contains another trusted pin for this request"
                )
            try:
                trusted_content_authentication = (
                    authenticate_reveal_store_trusted_stage_content_pin(
                        current,
                        current_tip,
                        reveal_request=request_value,
                        stage_access_manifest=access_manifest,
                    )
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Persisted trusted stage-content pin failed current-tip authentication"
                ) from exc
            parent_consumption_binding: dict[str, Any] | None = None
            if stage == "final":
                # Reload the exact parent objects from the post-pin current tip.
                # The pin-only revision is monotonic and state-preserving, but
                # this avoids carrying any pre-transition authorization object
                # into the verifier context.
                (
                    parent_entry,
                    parent_bundle,
                    parent_output_receipt,
                ) = _locate_parent_intermediate_predecessor(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    child_request=request_value,
                    prerequisite_stage_evidence=evidence,
                )
                parent_consumption_binding = _build_parent_consumption_binding(
                    authenticated_store_snapshot=current,
                    independent_current_tip_anchor=current_tip,
                    child_request=request_value,
                    parent_entry=parent_entry,
                    parent_bundle=parent_bundle,
                    parent_output_receipt=parent_output_receipt,
                    prerequisite_stage_evidence=evidence,
                )
            authenticated_store_context = _authenticated_store_verifier_context(
                trusted_stage_content_pin=trusted_content_pin,
                trusted_stage_content_authentication=trusted_content_authentication,
                parent_consumption_binding=parent_consumption_binding,
            )

            context_dict = _expected_context_for_consumed_request(
                request_value,
                trusted_stage_content_pin_sha256=trusted_content_pin["pin_sha256"],
                trusted_stage_content_authentication_receipt_sha256=(
                    trusted_content_authentication["authentication_receipt_sha256"]
                ),
                parent_consumption_binding_sha256=(
                    None
                    if parent_consumption_binding is None
                    else parent_consumption_binding[
                        "parent_consumption_binding_sha256"
                    ]
                ),
                authenticated_store_context_sha256=authenticated_store_context[
                    "authenticated_store_context_sha256"
                ],
            )
            context = MappingProxyType(context_dict)
            promotion_gate_before = AUTHORITATIVE_STAGE_PROMOTION_ENABLED
            restore_document = _restore_pending_document(
                state_bytes=original_state_bytes,
                tip_anchor_bytes=original_tip_anchor_bytes,
            )
            restore_document_bytes = _encoded_state(restore_document)
            # Persist the authenticated original pair before effectful verifier
            # code runs. A hard process stop after any verifier-side mutation
            # is therefore repaired on the next locked load.
            _atomic_replace(
                locked_restore_pending_path,
                restore_document_bytes,
            )
            try:
                try:
                    verifier_inputs = detach_untrusted_stage_json(
                        {
                            "evidence": evidence,
                            "stage_access_manifest": access_manifest,
                            "expected_context": context_dict,
                            "authenticated_store_context": authenticated_store_context,
                        },
                        "fixed verifier input bundle",
                    )
                    try:
                        result = authoritative_prerequisite_validator(
                            verifier_inputs["evidence"],
                            verifier_inputs["stage_access_manifest"],
                            verifier_inputs["expected_context"],
                            authenticated_store_context=verifier_inputs[
                                "authenticated_store_context"
                            ],
                        )
                    finally:
                        promotion_gate_changed = (
                            AUTHORITATIVE_STAGE_PROMOTION_ENABLED
                            is not promotion_gate_before
                        )
                        globals()["AUTHORITATIVE_STAGE_PROMOTION_ENABLED"] = (
                            promotion_gate_before
                        )
                except SecFilingGemmaRevealStoreError:
                    raise
                except Exception as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Fixed semantic prerequisite verifier failed; request was not consumed"
                    ) from exc
                if (
                    self._repository_root != locked_repository_root
                    or self._store_directory != locked_store_directory
                    or self.state_path != locked_state_path
                    or self.current_tip_anchor_path != locked_tip_anchor_path
                    or self.restore_pending_path != locked_restore_pending_path
                    or self.lock_path != locked_lock_path
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store paths changed during prerequisite validation"
                    )
                validation = _validate_semantic_result(
                    result,
                    expected_context=context,
                    expected_validator_id=AUTHORITATIVE_VALIDATOR_ID,
                    expected_validator_source_sha256=candidate_value["bindings"][
                        "source_hashes"
                    ]["stage_verifier"],
                )
                if promotion_gate_before is not True:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store stage promotion remains independently disabled"
                    )
                if promotion_gate_changed:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store stage promotion gate changed during validation"
                    )
                if (
                    self._repository_root != locked_repository_root
                    or self._store_directory != locked_store_directory
                    or self.state_path != locked_state_path
                    or self.current_tip_anchor_path != locked_tip_anchor_path
                    or self.restore_pending_path != locked_restore_pending_path
                    or self.lock_path != locked_lock_path
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store paths changed while validating the verifier result"
                    )
                # The verifier is effectful Python. Check exact bytes before
                # parsing so even a rehashed or differently encoded mutation is
                # rejected rather than silently replaced by the next state.
                try:
                    (
                        after_verifier,
                        after_verifier_tip,
                        after_verifier_bytes,
                        after_verifier_tip_bytes,
                    ) = self._read_state_and_tip_locked(
                        anchor,
                        recover_pending_restore=False,
                    )
                except SecFilingGemmaRevealStoreError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store state was damaged during prerequisite "
                        "validation; the prior authenticated state will be restored"
                    ) from exc
                if (
                    after_verifier_bytes != original_state_bytes
                    or after_verifier_tip_bytes != original_tip_anchor_bytes
                    or after_verifier != current
                    or after_verifier_tip != current_tip
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store state changed during prerequisite validation; "
                        "the prior authenticated state will be restored"
                    )
                persisted_restore = _read_regular_bytes(
                    locked_restore_pending_path,
                    "pre-verifier restore transaction",
                    max_bytes=MAX_RESTORE_PENDING_FILE_BYTES,
                )
                if persisted_restore != restore_document_bytes:
                    raise SecFilingGemmaRevealStoreError(
                        "Pre-verifier restore transaction changed during validation"
                    )
                locked_restore_pending_path.unlink()
                if (
                    locked_restore_pending_path.exists()
                    or locked_restore_pending_path.is_symlink()
                ):
                    raise SecFilingGemmaRevealStoreError(
                        "Pre-verifier restore transaction could not be retired"
                    )
            except BaseException:
                # Cleanup has finally-like semantics: verifier exceptions,
                # invalid semantic results, path redirection, and state-damage
                # errors cannot escape while mutated authoritative bytes remain.
                self._repository_root = locked_repository_root
                self._store_directory = locked_store_directory
                try:
                    restore_document = _restore_pending_document(
                        state_bytes=original_state_bytes,
                        tip_anchor_bytes=original_tip_anchor_bytes,
                    )
                    _atomic_replace(
                        locked_restore_pending_path,
                        _encoded_state(restore_document),
                    )
                    self._recover_pending_restore_locked(anchor)
                    restored_state_bytes = _read_regular_bytes(
                        locked_state_path,
                        "restored authoritative reveal-store state",
                        max_bytes=MAX_STATE_FILE_BYTES,
                    )
                    restored_tip_bytes = _read_regular_bytes(
                        locked_tip_anchor_path,
                        "restored independent current-tip anchor",
                        max_bytes=MAX_CURRENT_TIP_ANCHOR_FILE_BYTES,
                    )
                    if (
                        restored_state_bytes != original_state_bytes
                        or restored_tip_bytes != original_tip_anchor_bytes
                    ):
                        raise SecFilingGemmaRevealStoreError(
                            "Restored reveal-store bytes do not match the authenticated original"
                        )
                except BaseException as restore_exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store state could not be restored after prerequisite validation"
                    ) from restore_exc
                raise

            entries = copy.deepcopy(ledger["entries"])
            final_delta = 1 if stage == "final" else 0
            cumulative_final = (
                ledger["chain"]["actual_final_touch_count"] + final_delta
            )
            body = {
                "schema_version": CONSUMPTION_ENTRY_SCHEMA_VERSION,
                "sequence": len(entries) + 1,
                "request_sha256": request_hash,
                "request": request_value,
                "stage_access_manifest": access_manifest,
                "stage": stage,
                "attempt_id": request_value["attempt_id"],
                "candidate_sha256": request_value["candidate_sha256"],
                "registry_entry_sha256": request_value["registry_entry_sha256"],
                "prerequisite_validation": validation,
                "prior_tip_sha256": ledger["chain"]["tip_sha256"],
                "final_touch_delta": final_delta,
                "cumulative_actual_final_touch_count": cumulative_final,
            }
            entry = {**body, "entry_sha256": canonical_sha256(body)}
            entries.append(entry)
            next_ledger = _consumption_ledger(
                entries, tip_sha256=entry["entry_sha256"], anchor=anchor
            )
            next_state = _state_snapshot(
                anchor=anchor,
                latest_registry=current["latest_registry"],
                latest_pin=current["latest_registry_pin"],
                consumption_ledger=next_ledger,
            )
            bundle: dict[str, Any] | None = None
            if _issue_authorization_grant:
                try:
                    bundle = _authorization_bundle(
                        authenticated_store_snapshot=next_state,
                        expected_new_consumption_entry_sha256=entry[
                            "entry_sha256"
                        ],
                    )
                except SecFilingGemmaStageAuthorizationError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Authenticated consumption could not produce an exact authorization grant"
                    ) from exc

            # This transaction is intentionally not rolled back in an outer
            # exception handler.  If the process stops after the state replace,
            # the pending CAS anchor makes the exact state+bundle recoverable;
            # retrying the same request returns that stored bundle rather than
            # consuming a second entry.
            try:
                authenticated_next_state, authenticated_next_tip = (
                    self._commit_state_and_tip_locked(
                        tracked_anchor=anchor,
                        prior_tip_anchor=current_tip,
                        next_state=next_state,
                        authorization_bundle=bundle,
                    )
                )
            except SecFilingGemmaStageAuthorizationError as exc:
                raise SecFilingGemmaRevealStoreError(
                    "State/current-tip authorization transaction failed closed"
                ) from exc
            if bundle is None:
                return authenticated_next_state
            persisted = authenticated_next_tip["authorization_bundles"].get(
                request_hash
            )
            if persisted != bundle:
                raise SecFilingGemmaRevealStoreError(
                    "Committed current tip did not preserve the exact authorization bundle"
                )
            try:
                validate_consumed_stage_authorization_grant(
                    persisted["authorization_grant"],
                    authenticated_store_snapshot=authenticated_next_state,
                    external_store_state_pin=persisted["store_state_pin"],
                    independent_current_tip_anchor=authenticated_next_tip,
                    expected_consumption_entry_sha256=entry["entry_sha256"],
                    expected_request_sha256=request_hash,
                    expected_candidate_sha256=request_value["candidate_sha256"],
                    expected_stage=stage,
                    expected_prerequisite_stage_evidence_sha256=evidence_hash,
                    expected_stage_access_manifest_sha256=access_hash,
                    expected_output_namespace=access_manifest["output"]["namespace"],
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Committed authorization bundle failed independent current-tip validation"
                ) from exc
            return copy.deepcopy(persisted)

    def record_consumed_stage_output_evidence(
        self,
        *,
        request_sha256: str,
        stage_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist the first exact evidence candidate recorded for one current grant.

        The request-keyed receipt is a dedicated state-preserving CAS append.
        An exact retry returns the original receipt without another revision;
        a different second candidate for the same consumed grant is rejected.
        This binds caller-supplied evidence to grant issuance, but it does not
        attest that an authorized SEC/model/market reader produced those bytes.
        Production must keep promotion disabled until the owned runner is the
        only component permitted to call this method.
        """

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            tracked_anchor = _load_tracked_anchor(self.repository_root)
            current, current_tip, _state_bytes, _tip_bytes = (
                self._read_state_and_tip_locked(tracked_anchor)
            )
            request_hash = _sha256(
                request_sha256,
                "consumed-stage output request hash",
            )
            bundle = current_tip["authorization_bundles"].get(request_hash)
            if type(bundle) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Consumed-stage output lacks its persisted authorization bundle"
                )
            grant = _expect_mapping(
                bundle.get("authorization_grant"),
                "consumed-stage output authorization grant",
            )
            if type(stage_evidence) is not dict:
                raise SecFilingGemmaRevealStoreError(
                    "Consumed-stage output evidence must be an exact built-in dict"
                )
            try:
                detached_evidence = detach_untrusted_stage_json(
                    stage_evidence,
                    "consumed-stage output evidence",
                )
            except Exception as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Consumed-stage output evidence exceeds fixed allocation bounds"
                ) from exc
            binding = _stage_evidence_output_binding(
                detached_evidence,
                authorization_grant=grant,
            )
            try:
                validate_consumed_stage_authorization_grant(
                    grant,
                    authenticated_store_snapshot=current,
                    external_store_state_pin=bundle["store_state_pin"],
                    independent_current_tip_anchor=current_tip,
                    expected_consumption_entry_sha256=grant[
                        "consumption_entry_sha256"
                    ],
                    expected_request_sha256=request_hash,
                    expected_candidate_sha256=binding["output_candidate_sha256"],
                    expected_stage=binding[
                        "output_stage_evidence_prerequisite_stage"
                    ],
                    expected_prerequisite_stage_evidence_sha256=binding[
                        "output_parent_stage_evidence_sha256"
                    ],
                    expected_stage_access_manifest_sha256=grant[
                        "stage_access_manifest_sha256"
                    ],
                    expected_output_namespace=grant["output_namespace"],
                )
                receipt = build_consumed_stage_output_receipt(
                    bundle,
                    **binding,
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Consumed-stage output is not authorized by the exact current grant"
                ) from exc
            existing = current_tip["consumed_stage_output_receipts"].get(
                request_hash
            )
            if existing is not None:
                if existing != receipt:
                    raise SecFilingGemmaRevealStoreError(
                        "Consumed grant already has a different first output"
                    )
                try:
                    validate_consumed_stage_output_receipt(
                        existing,
                        authenticated_store_snapshot=current,
                        independent_current_tip_anchor=current_tip,
                        authorization_bundle=bundle,
                        **binding,
                    )
                except SecFilingGemmaStageAuthorizationError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Persisted consumed-stage output receipt is invalid"
                    ) from exc
                return copy.deepcopy(existing)
            committed_state, committed_tip = self._commit_state_and_tip_locked(
                tracked_anchor=tracked_anchor,
                prior_tip_anchor=current_tip,
                next_state=current,
                consumed_stage_output_receipt=receipt,
            )
            persisted = committed_tip["consumed_stage_output_receipts"].get(
                request_hash
            )
            try:
                validate_consumed_stage_output_receipt(
                    persisted,
                    authenticated_store_snapshot=committed_state,
                    independent_current_tip_anchor=committed_tip,
                    authorization_bundle=bundle,
                    **binding,
                )
            except (KeyError, SecFilingGemmaStageAuthorizationError) as exc:
                raise SecFilingGemmaRevealStoreError(
                    "Committed consumed-stage output receipt failed validation"
                ) from exc
            return copy.deepcopy(persisted)

    def consume_request_and_issue_authorization_grant(
        self,
        request: Mapping[str, Any],
        *,
        candidate_manifest: Mapping[str, Any],
        stage: str,
        stage_access_manifest: Mapping[str, Any],
        prerequisite_stage_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Consume one request and return its exact post-consumption grant bundle.

        The grant is derived for the newly appended ledger tip, and its exact
        bundle is committed in the separate monotonic current-tip anchor as the
        same recoverable transaction as the consumed state.  The bundled pin
        is not a trust root: a downstream API must independently load the
        store's current-tip anchor and pass it to grant validation.  The fixed
        production verifier still raises, so production cannot reach grant
        issuance while any prerequisite check remains unsupported.
        """

        return self.consume_request(
            request,
            candidate_manifest=candidate_manifest,
            stage=stage,
            stage_access_manifest=stage_access_manifest,
            prerequisite_stage_evidence=prerequisite_stage_evidence,
            _issue_authorization_grant=True,
        )


__all__ = [
    "AUTHORITATIVE_STAGE_PROMOTION_ENABLED",
    "AUTHORITATIVE_VALIDATOR_ID",
    "CONSUMPTION_ENTRY_SCHEMA_VERSION",
    "CONSUMPTION_LEDGER_SCHEMA_VERSION",
    "CURRENT_TIP_ANCHOR_FILENAME",
    "CURRENT_TIP_PENDING_SCHEMA_VERSION",
    "INITIAL_PIN_RELATIVE_PATH",
    "LOCK_FILENAME",
    "MAX_CURRENT_TIP_ANCHOR_FILE_BYTES",
    "MAX_RESTORE_PENDING_FILE_BYTES",
    "MAX_STATE_FILE_BYTES",
    "REQUIRED_SEMANTIC_CHECKS",
    "SEMANTIC_PREREQUISITE_SCHEMA_VERSION",
    "RESTORE_PENDING_FILENAME",
    "RESTORE_PENDING_SCHEMA_VERSION",
    "STATE_FILENAME",
    "STORE_SCHEMA_VERSION",
    "SecFilingGemmaRevealStore",
    "SecFilingGemmaRevealStoreError",
    "SemanticPrerequisiteValidation",
]
