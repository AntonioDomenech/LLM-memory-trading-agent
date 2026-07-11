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
import copy
from dataclasses import dataclass
import hmac
import json
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
    authoritative_prerequisite_validator,
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
_TEMP_RE = re.compile(
    rf"\.{re.escape(STATE_FILENAME)}\.[0-9a-f]{{32}}\.tmp\Z"
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
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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
    except (TypeError, ValueError) as exc:
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


def _read_regular_bytes(path: Path, location: str) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise SecFilingGemmaRevealStoreError(f"{location} does not exist") from exc
    _validate_regular_details(before, location)
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
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise SecFilingGemmaRevealStoreError(
                f"{location} identity changed while opening"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = path.lstat()
        _validate_regular_details(after, location)
        if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
            raise SecFilingGemmaRevealStoreError(
                f"{location} identity changed while reading"
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

    def _read_state_locked(self, anchor: Mapping[str, Any]) -> dict[str, Any]:
        try:
            payload = _read_regular_bytes(
                self.state_path, "authoritative reveal-store state"
            )
        except SecFilingGemmaRevealStoreError as exc:
            if not self.state_path.exists() and not self.state_path.is_symlink():
                raise SecFilingGemmaRevealStoreError(
                    "Reveal store is not initialized"
                ) from exc
            raise
        parsed = _strict_json_bytes(payload, "authoritative reveal-store state")
        return _validate_state(
            _expect_mapping(parsed, "authoritative reveal-store state"),
            expected_anchor=anchor,
        )

    def initialize(self) -> dict[str, Any]:
        """Create genesis state once, or validate and return existing state."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            if self.state_path.exists() or self.state_path.is_symlink():
                return self._read_state_locked(anchor)
            registry = build_initial_reveal_registry()
            pin = derive_registry_pin(registry)
            state = _state_snapshot(
                anchor=anchor,
                latest_registry=registry,
                latest_pin=pin,
                consumption_ledger=_initial_consumption_ledger(anchor),
            )
            _atomic_replace(self.state_path, _encoded_state(state))
            return self._read_state_locked(anchor)

    def load(self) -> dict[str, Any]:
        """Load and fully authenticate the latest state without mutating it."""

        with self._locked():
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            return self._read_state_locked(anchor)

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
            current = self._read_state_locked(anchor)
            transition_value = _json_value_copy(
                dict(_expect_mapping(transition, "registry transition")),
                "registry transition",
            )
            appended_value = _json_value_copy(
                dict(_expect_mapping(appended_registry, "appended registry")),
                "appended registry",
            )
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
            _atomic_replace(self.state_path, _encoded_state(next_state))
            return self._read_state_locked(anchor)

    def consume_request(
        self,
        request: Mapping[str, Any],
        *,
        candidate_manifest: Mapping[str, Any],
        stage: str,
        stage_access_manifest: Mapping[str, Any],
        prerequisite_stage_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate and atomically consume one non-authorizing stage request.

        The fixed candidate-bound verifier executes while the exclusive lock is
        held. It receives
        defensive read-only copies of the prerequisite evidence, the actual
        self-hashed stage-access manifest, and the expected request context.
        It must return :class:`SemanticPrerequisiteValidation`.  Failure,
        replay, verifier mutation of the state file, or any binding mismatch
        leaves no consumption entry.

        There is deliberately no caller-supplied validator parameter. Until
        the fixed verifier can complete every frozen check, it raises and this
        method cannot consume a request.
        """
        with self._locked():
            locked_repository_root = self._repository_root
            locked_store_directory = self._store_directory
            locked_state_path = self.state_path
            locked_lock_path = self.lock_path
            _cleanup_interrupted_temporaries(self.store_directory)
            anchor = _load_tracked_anchor(self.repository_root)
            current = self._read_state_locked(anchor)
            request_value = _json_value_copy(
                dict(_expect_mapping(request, "reveal request")),
                "reveal request",
            )
            candidate_value = _json_value_copy(
                dict(_expect_mapping(candidate_manifest, "candidate manifest")),
                "candidate manifest",
            )
            access_manifest = _json_value_copy(
                dict(_expect_mapping(stage_access_manifest, "stage access manifest")),
                "stage access manifest",
            )
            access_hash = _stage_access_manifest_sha256(access_manifest)
            evidence = _json_value_copy(
                dict(
                    _expect_mapping(
                        prerequisite_stage_evidence,
                        "prerequisite stage evidence",
                    )
                ),
                "prerequisite stage evidence",
            )
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
            if any(
                entry["request_sha256"] == request_hash
                or (
                    entry["attempt_id"] == request_value["attempt_id"]
                    and entry["stage"] == stage
                )
                for entry in ledger["entries"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Reveal request or candidate stage was already consumed"
                )
            candidate_key = (
                request_value["attempt_id"],
                request_value["candidate_sha256"],
                request_value["registry_entry_sha256"],
            )
            if stage == "final" and not any(
                entry["stage"] == "intermediate"
                and (
                    entry["attempt_id"],
                    entry["candidate_sha256"],
                    entry["registry_entry_sha256"],
                )
                == candidate_key
                for entry in ledger["entries"]
            ):
                raise SecFilingGemmaRevealStoreError(
                    "Final request requires prior intermediate-request consumption"
                )

            context_dict = {
                "prerequisite_stage": request_value["prerequisite_stage"],
                "prerequisite_stage_evidence_sha256": evidence_hash,
                "attempt_id": request_value["attempt_id"],
                "candidate_sha256": request_value["candidate_sha256"],
                "candidate_design_sha256": request_value[
                    "candidate_design_sha256"
                ],
                "registry_entry_sha256": request_value["registry_entry_sha256"],
                "request_sha256": request_hash,
                "stage": stage,
                "stage_access_manifest_sha256": request_value[
                    "stage_access_manifest_sha256"
                ],
                "registry_sha256": request_value["registry_sha256"],
                "registry_tip_sha256": request_value["registry_tip_sha256"],
            }
            context = MappingProxyType(context_dict)
            original_state_bytes = _read_regular_bytes(
                locked_state_path,
                "authoritative reveal-store state before prerequisite validation",
            )
            try:
                try:
                    result = authoritative_prerequisite_validator(
                        MappingProxyType(evidence),
                        MappingProxyType(access_manifest),
                        context,
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
                # The verifier is effectful Python. Check exact bytes before
                # parsing so even a rehashed or differently encoded mutation is
                # rejected rather than silently replaced by the next state.
                try:
                    after_verifier_bytes = _read_regular_bytes(
                        locked_state_path,
                        "authoritative reveal-store state after prerequisite validation",
                    )
                    after_verifier = self._read_state_locked(anchor)
                except SecFilingGemmaRevealStoreError as exc:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store state was damaged during prerequisite "
                        "validation; the prior authenticated state will be restored"
                    ) from exc
                if after_verifier_bytes != original_state_bytes or after_verifier != current:
                    raise SecFilingGemmaRevealStoreError(
                        "Reveal-store state changed during prerequisite validation; "
                        "the prior authenticated state will be restored"
                    )
            except BaseException:
                # Cleanup has finally-like semantics: verifier exceptions,
                # invalid semantic results, path redirection, and state-damage
                # errors cannot escape while mutated authoritative bytes remain.
                self._repository_root = locked_repository_root
                self._store_directory = locked_store_directory
                try:
                    _atomic_replace(locked_state_path, original_state_bytes)
                    restored_state_bytes = _read_regular_bytes(
                        locked_state_path,
                        "restored authoritative reveal-store state",
                    )
                    if restored_state_bytes != original_state_bytes:
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
            _atomic_replace(self.state_path, _encoded_state(next_state))
            return self._read_state_locked(anchor)


__all__ = [
    "AUTHORITATIVE_VALIDATOR_ID",
    "CONSUMPTION_ENTRY_SCHEMA_VERSION",
    "CONSUMPTION_LEDGER_SCHEMA_VERSION",
    "INITIAL_PIN_RELATIVE_PATH",
    "LOCK_FILENAME",
    "REQUIRED_SEMANTIC_CHECKS",
    "SEMANTIC_PREREQUISITE_SCHEMA_VERSION",
    "STATE_FILENAME",
    "STORE_SCHEMA_VERSION",
    "SecFilingGemmaRevealStore",
    "SecFilingGemmaRevealStoreError",
    "SemanticPrerequisiteValidation",
]
