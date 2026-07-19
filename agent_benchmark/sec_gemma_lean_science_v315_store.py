"""One-shot private attempt store for the v3.15 development experiment.

The store makes crash boundaries explicit.  It never performs an external
request: callers first persist an intent, perform one effect themselves, then
seal the response and a marker-last checkpoint through this API.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Final

from agent_benchmark.sec_gemma_lean_science_v315_journal import (
    HashChainJournal,
    JournalEvent,
    JournalReplay,
    V315JournalError,
    canonical_json_bytes,
    canonical_sha256,
    entry_exists_nofollow,
    fsync_directory,
    publish_noreplace,
    read_regular_bytes,
    regular_file,
    replay_journal,
    secure_directory,
    sha256_bytes,
    valid_sha256,
)


LOCK_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-attempt-lock-v1"
)
RESPONSE_PAYLOAD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-response-payload-v1"
)
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-checkpoint-v1"
)
RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-checkpoint-state-v1"
)
PAUSE_CHECKPOINT_STATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-pause-checkpoint-state-v2"
)
RUNTIME_BINDING_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-runtime-binding-receipt-v1"
)
TERMINAL_CANDIDATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-terminal-evidence-candidate-v1"
)
PAUSE_CANDIDATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-pause-publication-candidate-v1"
)
RECOVERY_INTENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-recovery-intent-v1"
)
RECOVERY_TERMINAL_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-recovery-terminal-v1"
)
# Kept as a compatibility export for callers that previously named the private
# terminal envelope.  V3.15 stores a terminal *candidate*, not V3.10 evidence.
TERMINAL_EVIDENCE_SCHEMA_VERSION: Final[str] = TERMINAL_CANDIDATE_SCHEMA_VERSION

ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-15-development-001"
)
PILOT_COUNT: Final[int] = 5
REMAINING_COUNT: Final[int] = 70
TOTAL_GENERATIONS: Final[int] = 75
YAHOO_REQUEST_COUNT: Final[int] = 6
YAHOO_RESPONSE_CAP_BYTES: Final[int] = 64 * 1024 * 1024
YAHOO_BATCH_BODY_CAP_BYTES: Final[int] = 128 * 1024 * 1024
MODEL_RESPONSE_CAP_BYTES: Final[int] = 256 * 1024
PAUSE_THRESHOLD_NS: Final[int] = 43_200_000_000_000

LOCK_FILENAME: Final[str] = ".attempt.lock"
JOURNAL_DIRECTORY: Final[str] = "journal"
PAYLOAD_DIRECTORY: Final[str] = "payloads"
CHECKPOINT_DIRECTORY: Final[str] = "checkpoints"
PUBLICATION_CANDIDATE_DIRECTORY: Final[str] = "publication_candidates"
RECOVERY_JOURNAL_DIRECTORY: Final[str] = "publication_recovery_journal"
TERMINAL_BUNDLE_DIRECTORY: Final[str] = "terminal"
PAUSE_BUNDLE_DIRECTORY: Final[str] = "pause"
TERMINAL_CANDIDATE_FILENAME: Final[str] = "candidate.json"
TERMINAL_PUBLIC_RESULT_FILENAME: Final[str] = "public_result.json"
TERMINAL_COMPARISON_BEFORE_FILENAME: Final[str] = "comparison_before.md"
TERMINAL_COMPARISON_AFTER_FILENAME: Final[str] = "comparison_after.md"
PAUSE_CANDIDATE_FILENAME: Final[str] = "candidate.json"
PAUSE_ARTIFACT_FILENAME: Final[str] = "artifact.json"
_ROOT_ENTRIES: Final[frozenset[str]] = frozenset(
    {
        LOCK_FILENAME,
        JOURNAL_DIRECTORY,
        PAYLOAD_DIRECTORY,
        CHECKPOINT_DIRECTORY,
        PUBLICATION_CANDIDATE_DIRECTORY,
        RECOVERY_JOURNAL_DIRECTORY,
    }
)
_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "plan",
        "attempt",
        "implementation",
        "preflight",
        "source",
        "science",
        "effect_budget",
        "request_order",
        "pilot_order",
    }
)
_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "attempt_intent",
        "request_intent",
        "response_committed",
        "checkpoint_committed",
        "paused_for_justification",
        "continuation_authorized",
        "market_values_opened",
        "model_responses_opened",
        "attempt_terminal",
        "runtime_binding_receipt",
        "runtime_binding_failed",
        "candidate_abandoned",
    }
)
_RECOVERY_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"recovery_intent", "runtime_binding_receipt", "recovery_terminal"}
)
_INVOCATION_KINDS: Final[frozenset[str]] = frozenset(
    {"development", "continuation", "publication_recovery"}
)
_RUNTIME_CHECKPOINTS: Final[frozenset[str]] = frozenset(
    {
        "attempt_open",
        *(f"yahoo_{index:02d}_pre" for index in range(1, 7)),
        "model_initial_pre",
        "model_initial_post",
        "pause_publish_pre",
        "model_continuation_pre",
        "model_continuation_post",
        "evaluation_pre",
        "evaluation_post",
        "result_publish_pre",
        "publication_recovery_pre",
    }
)
_RECOVERY_TARGET_KINDS: Final[frozenset[str]] = frozenset(
    {
        "pause",
        "terminal_bound",
        "terminal_demoted",
        "terminal_demoted_bound",
        "terminal_indeterminate_created",
        "zero_effect_binding_rejection",
    }
)
_EFFECT_KINDS: Final[frozenset[str]] = frozenset(
    {"yahoo", "identity", "gemma"}
)
_MODEL_PHASES: Final[frozenset[str]] = frozenset(
    {
        "none",
        "pre_probe_start",
        "pre_probe",
        "generation",
        "post_probe",
        "post_probe_close",
    }
)
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9._/-]{1,200}\Z")
_SAFE_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_CONTENT_NAME_RE = re.compile(r"([0-9a-f]{64})\.json\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_REPARSE_ATTRIBUTE: Final[int] = 0x400
_MAX_CONTENT_BYTES: Final[int] = 256 * 1024 * 1024


class V315StoreError(RuntimeError):
    """Fixed-code attempt-store rejection."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class V315StoreConflict(V315StoreError):
    """The attempt is already locked or an immutable destination exists."""


class V315StorePoisoned(V315StoreError):
    """The persisted attempt cannot be accepted or resumed."""


def _translate_journal_error(exc: V315JournalError) -> V315StorePoisoned:
    return V315StorePoisoned(exc.code)


def _validate_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(authority, Mapping):
        raise V315StoreError("authority_invalid")
    value = _thaw_json(authority)
    if frozenset(value) != _AUTHORITY_KEYS:
        raise V315StoreError("authority_invalid")
    attempt = value["attempt"]
    if not (
        attempt == ATTEMPT_ID
        or (
            type(attempt) is dict
            and attempt.get("attempt_id") == ATTEMPT_ID
        )
    ):
        raise V315StoreError("attempt_identity_invalid")
    try:
        # Round-trip once so tuples and mapping subclasses cannot create an
        # authority that writes canonically but compares differently on replay.
        normalized = json.loads(canonical_json_bytes(value))
    except (V315JournalError, json.JSONDecodeError):
        raise V315StoreError("authority_invalid") from None
    if type(normalized) is not dict:
        raise V315StoreError("authority_invalid")
    return normalized


def _safe_id(value: Any) -> bool:
    return type(value) is str and _SAFE_ID_RE.fullmatch(value) is not None


def _safe_code(value: Any) -> bool:
    return type(value) is str and _SAFE_CODE_RE.fullmatch(value) is not None


def _valid_commit(value: Any) -> bool:
    return type(value) is str and _COMMIT_RE.fullmatch(value) is not None


def _freeze_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Recursively copy immutable JSON containers into encoder-safe values.

    ``json.dumps`` does not serialize ``MappingProxyType`` even though the
    canonical validator correctly recognizes it as a mapping.  Store receipts
    deliberately freeze nested evidence, so every later serialization boundary
    must thaw the complete tree instead of applying a shallow ``dict`` copy.
    Tuples are normalized to lists at the same boundary, making the canonical
    representation independent of whether callers supplied mutable or frozen
    JSON containers.
    """

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_thaw_json(item) for item in value]
    return value


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def _mkdir_new(path: Path) -> None:
    try:
        path.mkdir()
        secure_directory(path)
        fsync_directory(path.parent)
    except V315JournalError as exc:
        raise _translate_journal_error(exc) from None
    except OSError:
        raise V315StorePoisoned("attempt_directory_creation_failed") from None


def _write_exclusive(path: Path, encoded: bytes, *, code: str) -> None:
    descriptor: int | None = None
    try:
        secure_directory(path.parent)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOINHERIT", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_reparse(opened)
            or opened.st_nlink != 1
        ):
            raise OSError("not_regular")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("short_write")
            offset += written
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        reread = bytearray()
        while len(reread) < len(encoded):
            chunk = os.read(descriptor, len(encoded) - len(reread))
            if not chunk:
                break
            reread.extend(chunk)
        after = os.fstat(descriptor)
        visible = regular_file(path, code=code)
        if (
            opened.st_dev != after.st_dev
            or opened.st_ino == 0
            or opened.st_ino != after.st_ino
            or opened.st_dev != visible.st_dev
            or opened.st_ino != visible.st_ino
            or after.st_size != len(encoded)
            or bytes(reread) != encoded
        ):
            raise OSError("changed")
    except V315JournalError as exc:
        raise V315StorePoisoned(exc.code) from None
    except OSError:
        raise V315StorePoisoned(code) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise V315StorePoisoned("attempt_root_invalid") from None
    return True


def _directory_details(path: Path, *, code: str) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError:
        raise V315StorePoisoned(code) from None
    if (
        not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
    ):
        raise V315StorePoisoned(code)
    return details


def _publish_attempt_root_noreplace(source: Path, destination: Path) -> None:
    """Atomically move a complete staging directory into an absent name.

    The POSIX branch deliberately refuses to fall back to plain ``rename``:
    plain rename may replace a concurrently inserted symlink or empty
    directory.  Linux ``renameat2(RENAME_NOREPLACE)`` and Darwin/BSD
    ``renameatx_np(RENAME_EXCL)`` provide the required atomic contract.
    Windows ``os.rename`` already fails when the destination exists.
    """

    if (
        not source.is_absolute()
        or not destination.is_absolute()
        or source.name in {"", ".", ".."}
        or destination.name in {"", ".", ".."}
    ):
        raise V315StorePoisoned("attempt_initialization_failed")
    secure_directory(source.parent)
    secure_directory(destination.parent)
    source_details = _directory_details(
        source, code="attempt_initialization_failed"
    )
    if _entry_exists(destination):
        raise V315StoreConflict("attempt_already_exists")

    source_parent_descriptor: int | None = None
    destination_parent_descriptor: int | None = None
    try:
        if os.name == "nt":
            # MoveFile semantics used by os.rename on Windows do not replace
            # an existing destination directory or reparse point.
            os.rename(source, destination)
        else:
            import ctypes

            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            source_parent_descriptor = os.open(
                source.parent, directory_flags
            )
            destination_parent_descriptor = os.open(
                destination.parent, directory_flags
            )
            for descriptor in (
                source_parent_descriptor,
                destination_parent_descriptor,
            ):
                parent_details = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(parent_details.st_mode)
                    or _is_reparse(parent_details)
                ):
                    raise OSError("parent_changed")

            libc = ctypes.CDLL(None, use_errno=True)
            source_name = os.fsencode(source.name)
            destination_name = os.fsencode(destination.name)
            renameat2 = getattr(libc, "renameat2", None)
            if renameat2 is not None:
                renameat2.argtypes = (
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                )
                renameat2.restype = ctypes.c_int
                result = renameat2(
                    source_parent_descriptor,
                    source_name,
                    destination_parent_descriptor,
                    destination_name,
                    1,  # RENAME_NOREPLACE
                )
            else:
                renameatx_np = getattr(libc, "renameatx_np", None)
                if renameatx_np is None:
                    raise OSError("atomic_noreplace_unavailable")
                renameatx_np.argtypes = (
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                )
                renameatx_np.restype = ctypes.c_int
                result = renameatx_np(
                    source_parent_descriptor,
                    source_name,
                    destination_parent_descriptor,
                    destination_name,
                    4,  # RENAME_EXCL
                )
            if result != 0:
                error_number = ctypes.get_errno()
                raise OSError(error_number, "atomic_noreplace_failed")
    except V315StoreError:
        raise
    except (OSError, V315JournalError):
        if _entry_exists(destination):
            raise V315StoreConflict("attempt_already_exists") from None
        raise V315StorePoisoned("attempt_initialization_failed") from None
    finally:
        if source_parent_descriptor is not None:
            os.close(source_parent_descriptor)
        if destination_parent_descriptor is not None:
            os.close(destination_parent_descriptor)

    published = _directory_details(
        destination, code="attempt_initialization_failed"
    )
    if (
        published.st_dev != source_details.st_dev
        or source_details.st_ino == 0
        or published.st_ino != source_details.st_ino
        or _entry_exists(source)
    ):
        raise V315StorePoisoned("attempt_initialization_failed")


def _initialize_attempt_root(
    root: Path,
    *,
    authority: Mapping[str, Any],
    authority_sha256: str,
    lock_bytes: bytes,
) -> None:
    """Build intent-complete state off-path, then publish the root atomically.

    A crash while constructing the staging tree cannot expose a partial
    authoritative ``development/`` directory.  Staging trees contain no
    response or contact data and are deliberately left untouched after an
    interruption; a later invocation creates a fresh isolated tree.
    """

    try:
        secure_directory(root.parent)
        if _entry_exists(root):
            raise V315StoreConflict("attempt_already_exists")
        staging_parent = secure_directory(root.parent.parent)
        staging = Path(
            tempfile.mkdtemp(
                prefix=".v315-attempt-initialization-",
                dir=str(staging_parent),
            )
        ).resolve(strict=True)
        staging.relative_to(staging_parent)
        secure_directory(staging)
        _mkdir_new(staging / JOURNAL_DIRECTORY)
        _mkdir_new(staging / PAYLOAD_DIRECTORY)
        _mkdir_new(staging / CHECKPOINT_DIRECTORY)
        _mkdir_new(staging / PUBLICATION_CANDIDATE_DIRECTORY)
        _mkdir_new(staging / RECOVERY_JOURNAL_DIRECTORY)
        _write_exclusive(
            staging / LOCK_FILENAME,
            lock_bytes,
            code="attempt_lock_creation_failed",
        )
        journal = HashChainJournal(
            staging / JOURNAL_DIRECTORY,
            authority_sha256=authority_sha256,
        )
        journal.append("attempt_intent", {"authority": dict(authority)})
        _root_layout(staging)
        replay = replay_journal(
            staging / JOURNAL_DIRECTORY,
            authority_sha256=authority_sha256,
        )
        machine = _replay_state(replay, authority)
        if len(machine.events) != 1 or machine.events[0].event_type != "attempt_intent":
            raise V315StorePoisoned("attempt_initialization_invalid")
        fsync_directory(staging)
        fsync_directory(staging_parent)
        if _entry_exists(root):
            raise V315StoreConflict("attempt_already_exists")
        _publish_attempt_root_noreplace(staging, root)
        fsync_directory(root.parent)
        fsync_directory(staging_parent)
    except (V315StoreError, V315JournalError):
        raise
    except (OSError, ValueError):
        if _entry_exists(root):
            raise V315StoreConflict("attempt_already_exists") from None
        raise V315StorePoisoned("attempt_initialization_failed") from None


_ACTIVE_LOCKS: set[Path] = set()
_ACTIVE_LOCKS_GUARD = threading.Lock()


class _AttemptLock:
    def __init__(self, path: Path, expected_bytes: bytes) -> None:
        self.path = path
        self.expected_bytes = expected_bytes
        self._handle: Any = None

    def acquire(self) -> None:
        with _ACTIVE_LOCKS_GUARD:
            if self.path in _ACTIVE_LOCKS:
                raise V315StoreConflict("attempt_locked")
            _ACTIVE_LOCKS.add(self.path)
        handle: Any = None
        try:
            before = regular_file(self.path, code="attempt_lock_invalid")
            handle = self.path.open("r+b", buffering=0)
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            after = os.fstat(handle.fileno())
            current = self.path.lstat()
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or after.st_dev != current.st_dev
                or after.st_ino != current.st_ino
                or not stat.S_ISREG(current.st_mode)
                or stat.S_ISLNK(current.st_mode)
                or _is_reparse(current)
                or current.st_nlink != 1
            ):
                raise V315StorePoisoned("attempt_lock_identity_changed")
            handle.seek(0)
            if handle.read() != self.expected_bytes:
                raise V315StorePoisoned("attempt_lock_invalid")
            self._handle = handle
        except Exception as exc:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            with _ACTIVE_LOCKS_GUARD:
                _ACTIVE_LOCKS.discard(self.path)
            if isinstance(exc, V315StoreError):
                raise
            if isinstance(exc, V315JournalError):
                raise _translate_journal_error(exc) from None
            raise V315StoreConflict("attempt_locked") from None

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None
            with _ACTIVE_LOCKS_GUARD:
                _ACTIVE_LOCKS.discard(self.path)


@dataclass(frozen=True)
class RequestIntent:
    event_sha256: str
    request_id: str
    effect_kind: str
    request_sha256: str
    ordinal: int
    segment_id: str | None
    model_phase: str


@dataclass(frozen=True)
class CommittedResponse:
    event_sha256: str
    intent: RequestIntent
    payload_sha256: str
    payload_bytes: int
    body_sha256: str
    body_bytes: int
    duration_ns: int | None
    checkpoint_event_sha256: str | None
    checkpoint_sha256: str | None
    body: bytes = field(repr=False)
    metadata: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class CommittedResponseReceipt:
    """Opaque response provenance safe to use without opening its body."""

    event_sha256: str
    intent: RequestIntent
    payload_sha256: str
    payload_bytes: int
    body_sha256: str
    body_bytes: int
    duration_ns: int | None
    checkpoint_event_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class PendingResponseReceipt:
    """Body-opaque receipt for deterministic checkpoint reconstruction."""

    event_sha256: str
    intent: RequestIntent
    payload_sha256: str
    payload_bytes: int
    body_sha256: str
    body_bytes: int
    duration_ns: int | None


@dataclass(frozen=True)
class RuntimeBindingReceipt:
    event_sha256: str
    checkpoint_name: str
    invocation_kind: str
    runtime_binding_receipt_sha256: str
    previous_runtime_binding_receipt_sha256: str | None
    receipt: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class TerminalCandidateReceipt:
    candidate_sha256: str
    candidate_bytes: int
    journal_head_before_candidate_sha256: str
    proposed_status: str
    proposed_terminal_code: str
    terminal_material_sha256: str
    public_result_sha256: str
    public_result_bytes: int
    comparison_before_sha256: str
    comparison_after_sha256: str
    comparison_after_bytes: int
    evidence: Mapping[str, Any] = field(repr=False)
    candidate: Mapping[str, Any] = field(repr=False)
    public_result: bytes = field(repr=False)
    comparison_before: bytes = field(repr=False)
    comparison_after: bytes = field(repr=False)


@dataclass(frozen=True)
class PauseCandidateReceipt:
    pause_candidate_sha256: str
    candidate_bytes: int
    journal_head_before_candidate_sha256: str
    pause_artifact_sha256: str
    pause_artifact_bytes: int
    candidate: Mapping[str, Any] = field(repr=False)
    pause_artifact: bytes = field(repr=False)


@dataclass(frozen=True)
class TerminalEvidenceReceipt:
    """Replay-audited terminal event plus its immutable V3.15 candidate."""

    event_sha256: str
    candidate_sha256: str
    candidate_bytes: int
    journal_head_before_terminal_sha256: str
    status: str
    terminal_code: str
    terminal_material_sha256: str
    public_result_sha256: str
    public_result_bytes: int
    comparison_before_sha256: str
    comparison_after_sha256: str
    comparison_after_bytes: int
    evidence: Mapping[str, Any] = field(repr=False)
    public_result: bytes = field(repr=False)
    comparison_before: bytes = field(repr=False)
    comparison_after: bytes = field(repr=False)

    @property
    def evidence_sha256(self) -> str:
        return self.candidate_sha256

    @property
    def evidence_bytes(self) -> int:
        return self.candidate_bytes


@dataclass(frozen=True)
class RecoveryIntentReceipt:
    event_sha256: str
    ordinal: int
    target_kind: str
    recovery_intent_sha256: str
    candidate_sha256: str
    attempt_terminal_event_sha256: str | None
    previous_recovery_terminal_sha256: str | None
    payload: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class RecoveryTerminalReceipt:
    event_sha256: str
    ordinal: int
    target_kind: str
    recovery_terminal_sha256: str
    outcome: str
    final_state_sha256: str
    payload: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True)
class SegmentSummary:
    segment_id: str
    generation_count: int
    identity_response_count: int
    generation_durations_ns: tuple[int, ...]
    close_response_event_sha256: str


@dataclass(frozen=True)
class AttemptSnapshot:
    authority_sha256: str
    journal_head_sha256: str
    event_count: int
    status: str
    terminal_code: str | None
    request_intent_count: int
    yahoo_intent_count: int
    identity_intent_count: int
    gemma_intent_count: int
    response_count: int
    checkpoint_count: int
    yahoo_response_count: int
    yahoo_body_bytes: int
    identity_response_count: int
    gemma_response_count: int
    active_request_id: str | None
    pending_checkpoint_event_sha256: str | None
    open_model_segment_id: str | None
    segment_summaries: tuple[SegmentSummary, ...]
    pause_required: bool
    paused: bool
    continuation_authorized: bool
    continuation_sha256: str | None
    continuation_permission_sha256: str | None
    continuation_commit: str | None
    market_values_opened: bool
    model_responses_opened: bool
    recoverable_checkpoint: bool


@dataclass(frozen=True)
class PendingCheckpointContext:
    subject_event_sha256: str
    subject_type: str
    response: PendingResponseReceipt | None
    pause_payload: Mapping[str, Any] | None
    events: tuple[JournalEvent, ...] = field(repr=False)


@dataclass
class _Machine:
    authority_sha256: str
    events: tuple[JournalEvent, ...]
    intents: dict[str, RequestIntent]
    intent_ids: set[str]
    responses: dict[str, JournalEvent]
    checkpoints: dict[str, JournalEvent]
    active_intent: RequestIntent | None
    pending_subject_event_sha256: str | None
    pending_subject_type: str | None
    yahoo_responses: int
    yahoo_body_bytes: int
    identity_responses: int
    gemma_responses: int
    open_segment_id: str | None
    segment_phase: str | None
    segment_pre_count: int
    segment_post_count: int
    segment_durations: list[int]
    segments: list[SegmentSummary]
    pause_event: JournalEvent | None
    pause_committed: bool
    continuation_event: JournalEvent | None
    market_values_opened: bool
    model_responses_opened: bool
    terminal_event: JournalEvent | None
    terminal_code: str | None
    runtime_receipts: list[RuntimeBindingReceipt]
    runtime_binding_failed: JournalEvent | None
    candidate_abandoned: JournalEvent | None


def _new_machine(authority_sha256: str, events: tuple[JournalEvent, ...]) -> _Machine:
    return _Machine(
        authority_sha256,
        events,
        {},
        set(),
        {},
        {},
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        None,
        None,
        0,
        0,
        [],
        [],
        None,
        False,
        None,
        False,
        False,
        None,
        None,
        [],
        None,
        None,
    )


def _exact_keys(payload: Mapping[str, Any], keys: set[str]) -> bool:
    return type(payload) is dict and set(payload) == keys


_RUNTIME_RECEIPT_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "checkpoint_name",
    "invocation_kind",
    "head",
    "tree",
    "clean_state_sha256",
    "repository_runtime_manifest_sha256",
    "execution_dependency_manifest_sha256",
    "loaded_code_manifest_sha256",
    "runtime_binding_authority_sha256",
    "route_argv_sha256",
    "process_environment_sha256",
    "interpreter_identity_sha256",
    "previous_runtime_binding_receipt_sha256",
    "runtime_binding_receipt_sha256",
)


def _payload_self_hash(payload: Mapping[str, Any], field: str) -> str:
    if type(payload) is not dict or field not in payload:
        raise V315StorePoisoned("self_hash_invalid")
    expected = canonical_sha256({key: value for key, value in payload.items() if key != field})
    value = payload[field]
    if not valid_sha256(value) or value != expected:
        raise V315StorePoisoned("self_hash_invalid")
    return value


def _decode_runtime_binding_receipt(
    event: JournalEvent,
) -> RuntimeBindingReceipt:
    payload = event.payload
    if not _exact_keys(payload, set(_RUNTIME_RECEIPT_KEYS)):
        raise V315StorePoisoned("runtime_binding_receipt_invalid")
    if (
        payload["schema_version"] != RUNTIME_BINDING_RECEIPT_SCHEMA_VERSION
        or payload["checkpoint_name"] not in _RUNTIME_CHECKPOINTS
        or payload["invocation_kind"] not in _INVOCATION_KINDS
        or not _valid_commit(payload["head"])
        or not _valid_commit(payload["tree"])
        or any(
            not valid_sha256(payload[key])
            for key in (
                "clean_state_sha256",
                "repository_runtime_manifest_sha256",
                "execution_dependency_manifest_sha256",
                "loaded_code_manifest_sha256",
                "runtime_binding_authority_sha256",
                "route_argv_sha256",
                "process_environment_sha256",
                "interpreter_identity_sha256",
            )
        )
        or (
            payload["previous_runtime_binding_receipt_sha256"] is not None
            and not valid_sha256(
                payload["previous_runtime_binding_receipt_sha256"]
            )
        )
    ):
        raise V315StorePoisoned("runtime_binding_receipt_invalid")
    try:
        digest = _payload_self_hash(payload, "runtime_binding_receipt_sha256")
    except V315StorePoisoned:
        raise V315StorePoisoned("runtime_binding_receipt_invalid") from None
    return RuntimeBindingReceipt(
        event.event_sha256,
        payload["checkpoint_name"],
        payload["invocation_kind"],
        digest,
        payload["previous_runtime_binding_receipt_sha256"],
        _freeze_json(dict(payload)),
    )


@dataclass
class _RecoveryMachine:
    events: tuple[JournalEvent, ...]
    intents: list[RecoveryIntentReceipt]
    runtime_receipts: list[RuntimeBindingReceipt]
    terminals: list[RecoveryTerminalReceipt]
    phase: str


def _greatest_recovery_terminal_sha256(
    terminals: list[RecoveryTerminalReceipt],
) -> str | None:
    return terminals[-1].recovery_terminal_sha256 if terminals else None


def _attempt_events_by_sha(machine: _Machine) -> dict[str, JournalEvent]:
    return {event.event_sha256: event for event in machine.events}


def _attempt_receipt_events(
    machine: _Machine,
) -> dict[str, JournalEvent]:
    events = _attempt_events_by_sha(machine)
    result: dict[str, JournalEvent] = {}
    for receipt in machine.runtime_receipts:
        event = events.get(receipt.event_sha256)
        if event is None:
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        result[receipt.runtime_binding_receipt_sha256] = event
    return result


def _recovery_intent_values(
    intent: RecoveryIntentReceipt,
) -> tuple[str, int, str, str]:
    payload = intent.payload
    return (
        payload["public_artifact_sha256"],
        payload["public_artifact_bytes"],
        payload["comparison_before_sha256"],
        payload["comparison_after_sha256"],
    )


def _model_post_binding_missing(machine: _Machine) -> bool:
    """Say whether a durably closed model segment lacks its exact post bind."""

    receipt_events = _attempt_receipt_events(machine)
    for index, segment in enumerate(machine.segments):
        checkpoint = machine.checkpoints.get(
            segment.close_response_event_sha256
        )
        if checkpoint is None:
            # The inherited deterministic checkpoint is recovered first.  It
            # is not a post-binding failure until that marker is durable.
            continue
        expected_name = (
            "model_initial_post" if index == 0 else "model_continuation_post"
        )
        matched = any(
            receipt.checkpoint_name == expected_name
            and receipt_events[receipt.runtime_binding_receipt_sha256]
            .previous_event_sha256
            == checkpoint.event_sha256
            for receipt in machine.runtime_receipts
        )
        if not matched:
            return True
    return False


def _candidate_less_terminal_code(machine: _Machine) -> str | None:
    """Apply the preregistered irreversible-ambiguity code precedence."""

    if machine.active_intent is not None:
        return "request_outcome_unknown"
    if _model_post_binding_missing(machine):
        return "model_post_binding_missing"
    if (
        machine.runtime_binding_failed is not None
        and machine.runtime_binding_failed.payload["external_intent_count"] > 0
    ):
        return "post_intent_runtime_binding_failed"
    if machine.open_segment_id is not None:
        return "irreversible_state_ambiguous"
    if machine.pending_subject_event_sha256 is not None:
        if machine.pending_subject_type != "response":
            return None
        response = machine.responses.get(machine.pending_subject_event_sha256)
        if response is None:
            return "irreversible_state_ambiguous"
        intent = machine.intents.get(
            response.payload["request_intent_event_sha256"]
        )
        if intent is None:
            return "irreversible_state_ambiguous"
        if intent.effect_kind == "yahoo" or intent.model_phase == "post_probe_close":
            return None
        return "irreversible_state_ambiguous"
    return None


def _attempt_receipt_at_head(
    machine: _Machine,
) -> RuntimeBindingReceipt | None:
    if not machine.runtime_receipts:
        return None
    receipt = machine.runtime_receipts[-1]
    if receipt.event_sha256 != machine.events[-1].event_sha256:
        return None
    return receipt


def _fresh_process_boundary_safe(machine: _Machine) -> bool:
    """Return whether replay permits a new process-open/authentication prefix.

    A completed, checkpointed request is not ambiguous.  An outstanding
    request, an open model identity epoch, an uncommitted checkpoint, or any
    terminal binding state is.  The fresh process marker is therefore allowed
    only at a replay-proven local boundary; it can never make an uncertain
    external effect retryable.
    """

    return (
        machine.terminal_event is None
        and machine.runtime_binding_failed is None
        and machine.candidate_abandoned is None
        and machine.active_intent is None
        and machine.pending_subject_event_sha256 is None
        and machine.open_segment_id is None
        and not _model_post_binding_missing(machine)
    )


def _exceptional_candidate_outcome(
    machine: _Machine,
) -> tuple[str, str, str] | None:
    failure = machine.runtime_binding_failed
    if failure is not None and failure.payload["external_intent_count"] == 0:
        return (
            "rejected",
            "runtime_binding_mismatch",
            "zero_effect_binding_rejection",
        )
    code = _candidate_less_terminal_code(machine)
    if code is not None:
        return (
            "indeterminate",
            code,
            "terminal_indeterminate_created",
        )
    return None


def _projected_ns(machine: _Machine) -> int | None:
    if machine.segments:
        durations = machine.segments[0].generation_durations_ns
    else:
        durations = tuple(machine.segment_durations)
    if len(durations) < PILOT_COUNT:
        return None
    pilots = durations[:PILOT_COUNT]
    return sum(pilots) + REMAINING_COUNT * max(pilots)


def _derive_response_checkpoint_state(
    machine: _Machine, subject_event_sha256: str
) -> dict[str, Any]:
    response = machine.responses.get(subject_event_sha256)
    if response is None:
        raise V315StorePoisoned("checkpoint_subject_invalid")
    intent = machine.intents.get(
        response.payload["request_intent_event_sha256"]
    )
    if intent is None:
        raise V315StorePoisoned("checkpoint_subject_invalid")
    return {
        "schema_version": RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION,
        "phase": intent.model_phase,
        "request_sha256": intent.request_sha256,
        "body_sha256": response.payload["body_sha256"],
    }


def _derive_pause_checkpoint_state(
    machine: _Machine, subject_event_sha256: str
) -> dict[str, Any]:
    pause = machine.pause_event
    if (
        pause is None
        or pause.event_sha256 != subject_event_sha256
        or not machine.segments
    ):
        raise V315StorePoisoned("checkpoint_subject_invalid")
    segment = machine.segments[0]
    durations = tuple(segment.generation_durations_ns)
    if (
        segment.generation_count != PILOT_COUNT
        or len(durations) != PILOT_COUNT
        or any(type(item) is not int or item <= 0 for item in durations)
    ):
        raise V315StorePoisoned("pause_checkpoint_state_invalid")
    projected = sum(durations) + REMAINING_COUNT * max(durations)
    prior = tuple(
        event for event in machine.events if event.sequence < pause.sequence
    )
    response_count = sum(
        event.event_type == "response_committed" for event in prior
    )
    checkpoint_count = sum(
        event.event_type == "checkpoint_committed" for event in prior
    )
    request_count = sum(
        event.event_type == "request_intent" for event in prior
    )
    pilot_responses = tuple(
        event
        for event in prior
        if event.event_type == "response_committed"
        and machine.intents[
            event.payload["request_intent_event_sha256"]
        ].effect_kind
        == "gemma"
        and machine.intents[
            event.payload["request_intent_event_sha256"]
        ].segment_id
        == segment.segment_id
    )
    close_checkpoint = machine.checkpoints.get(
        segment.close_response_event_sha256
    )
    pilot_checkpoints = tuple(
        machine.checkpoints.get(event.event_sha256)
        for event in pilot_responses
    )
    if (
        projected <= PAUSE_THRESHOLD_NS
        or pause.payload["pilot_count"] != PILOT_COUNT
        or pause.payload["pilot_durations_ns"] != list(durations)
        or pause.payload["projected_ns"] != projected
        or pause.payload["threshold_ns"] != PAUSE_THRESHOLD_NS
        or pause.payload["formula"]
        != "sum(pilot_duration_ns)+70*max(pilot_duration_ns)"
        or request_count != YAHOO_REQUEST_COUNT + 4 + PILOT_COUNT
        or response_count != YAHOO_REQUEST_COUNT + 4 + PILOT_COUNT
        or checkpoint_count != YAHOO_REQUEST_COUNT + 4 + PILOT_COUNT
        or len(pilot_responses) != PILOT_COUNT
        or close_checkpoint is None
        or any(item is None for item in pilot_checkpoints)
    ):
        raise V315StorePoisoned("pause_checkpoint_state_invalid")
    return {
        "schema_version": PAUSE_CHECKPOINT_STATE_SCHEMA_VERSION,
        "phase": "paused_for_justification",
        "segment_id": segment.segment_id,
        "pilot_count": PILOT_COUNT,
        "pilot_durations_ns": list(durations),
        "projected_ns": projected,
        "threshold_ns": PAUSE_THRESHOLD_NS,
        "formula": "sum(pilot_duration_ns)+70*max(pilot_duration_ns)",
        "pause_candidate_sha256": pause.payload["pause_candidate_sha256"],
        "pause_artifact_sha256": pause.payload["pause_artifact_sha256"],
        "pause_artifact_bytes": pause.payload["pause_artifact_bytes"],
        "publication_binding_receipt_sha256": pause.payload[
            "publication_binding_receipt_sha256"
        ],
        "journal_head_before_pause_sha256": pause.previous_event_sha256,
        "segment_close_response_event_sha256": (
            segment.close_response_event_sha256
        ),
        "segment_close_checkpoint_event_sha256": (
            close_checkpoint.event_sha256
        ),
        "pilot_response_event_sha256s": [
            event.event_sha256 for event in pilot_responses
        ],
        "pilot_checkpoint_event_sha256s": [
            item.event_sha256 for item in pilot_checkpoints if item is not None
        ],
    }


def _derive_checkpoint_state(
    machine: _Machine,
    subject_event_sha256: str,
    subject_type: str,
) -> dict[str, Any]:
    if subject_type == "response":
        return _derive_response_checkpoint_state(
            machine, subject_event_sha256
        )
    if subject_type == "pause":
        return _derive_pause_checkpoint_state(machine, subject_event_sha256)
    raise V315StorePoisoned("checkpoint_subject_invalid")


def _validate_request_transition(
    machine: _Machine,
    intent: RequestIntent,
    *,
    previous_event_sha256: str,
) -> None:
    if machine.terminal_event is not None:
        raise V315StorePoisoned("event_after_terminal")
    if machine.active_intent is not None or machine.pending_subject_event_sha256:
        raise V315StorePoisoned("request_boundary_incomplete")
    if machine.pause_committed and machine.continuation_event is None:
        raise V315StorePoisoned("request_while_paused")
    if intent.ordinal != len(machine.intents) + 1:
        raise V315StorePoisoned("request_order_invalid")
    if intent.request_id in machine.intent_ids:
        raise V315StorePoisoned("duplicate_request")

    if intent.effect_kind == "yahoo":
        if (
            intent.segment_id is not None
            or intent.model_phase != "none"
            or machine.open_segment_id is not None
            or machine.segments
            or machine.yahoo_responses >= YAHOO_REQUEST_COUNT
        ):
            raise V315StorePoisoned("yahoo_order_invalid")
        return

    if intent.effect_kind == "gemma":
        if (
            intent.model_phase != "generation"
            or intent.segment_id != machine.open_segment_id
            or machine.segment_phase != "generation"
            or machine.segment_pre_count != 2
        ):
            raise V315StorePoisoned("model_order_invalid")
        current = len(machine.segment_durations)
        expected = TOTAL_GENERATIONS if not machine.segments else REMAINING_COUNT
        if current >= expected:
            raise V315StorePoisoned("generation_count_exceeded")
        if not machine.segments and current == PILOT_COUNT:
            projection = _projected_ns(machine)
            if projection is None:
                raise V315StorePoisoned("pilot_timing_invalid")
            if projection > PAUSE_THRESHOLD_NS:
                raise V315StoreError("pilot_pause_required")
        return

    # Identity requests are the only effects that can open and close a model
    # segment.  Each probe is exactly version then show: two responses.
    if intent.effect_kind != "identity" or intent.segment_id is None:
        raise V315StorePoisoned("model_order_invalid")
    phase = intent.model_phase
    if phase == "pre_probe_start":
        continuation_pre = next(
            (
                item
                for item in reversed(machine.runtime_receipts)
                if item.event_sha256 == previous_event_sha256
            ),
            None,
        )
        if (
            machine.yahoo_responses != YAHOO_REQUEST_COUNT
            or machine.open_segment_id is not None
            or len(machine.segments) > 1
            or any(s.segment_id == intent.segment_id for s in machine.segments)
            or (machine.segments and machine.continuation_event is None)
            or (not machine.segments and machine.continuation_event is not None)
            or machine.segments
            and (
                continuation_pre is None
                or continuation_pre.invocation_kind != "continuation"
                or continuation_pre.checkpoint_name != "model_continuation_pre"
            )
        ):
            raise V315StorePoisoned("model_segment_start_invalid")
        return
    if intent.segment_id != machine.open_segment_id:
        raise V315StorePoisoned("model_segment_mismatch")
    if phase == "pre_probe":
        if machine.segment_phase != "pre" or machine.segment_pre_count != 1:
            raise V315StorePoisoned("pre_probe_order_invalid")
        return
    if phase == "post_probe":
        if machine.segment_phase != "generation" or machine.segment_post_count:
            raise V315StorePoisoned("post_probe_order_invalid")
        count = len(machine.segment_durations)
        if not machine.segments:
            projection = _projected_ns(machine)
            expected = (
                PILOT_COUNT
                if projection is not None and projection > PAUSE_THRESHOLD_NS
                else TOTAL_GENERATIONS
            )
        else:
            expected = REMAINING_COUNT
        if count != expected:
            raise V315StorePoisoned("post_probe_generation_count_invalid")
        return
    if phase == "post_probe_close":
        if machine.segment_phase != "post" or machine.segment_post_count != 1:
            raise V315StorePoisoned("post_probe_order_invalid")
        return
    raise V315StorePoisoned("model_phase_invalid")


def _private_values_ready(machine: _Machine) -> bool:
    layout = [item.generation_count for item in machine.segments]
    expected_identity = 4 if layout == [TOTAL_GENERATIONS] else 8
    return (
        machine.active_intent is None
        and machine.pending_subject_event_sha256 is None
        and machine.open_segment_id is None
        and machine.yahoo_responses == YAHOO_REQUEST_COUNT
        and machine.gemma_responses == TOTAL_GENERATIONS
        and machine.identity_responses == expected_identity
        and layout in ([TOTAL_GENERATIONS], [PILOT_COUNT, REMAINING_COUNT])
        and (not machine.pause_committed or machine.continuation_event is not None)
    )


def _validate_private_open_transition(machine: _Machine, event_type: str) -> None:
    if not _private_values_ready(machine):
        raise V315StorePoisoned("private_values_open_order_invalid")
    if event_type == "market_values_opened":
        if machine.market_values_opened or machine.model_responses_opened:
            raise V315StorePoisoned("private_values_open_order_invalid")
        return
    if event_type == "model_responses_opened":
        if not machine.market_values_opened or machine.model_responses_opened:
            raise V315StorePoisoned("private_values_open_order_invalid")
        return
    raise V315StorePoisoned("private_values_open_order_invalid")


def _validate_attempt_receipt_transition(
    machine: _Machine,
    event: JournalEvent,
    receipt: RuntimeBindingReceipt,
) -> None:
    checkpoint = receipt.checkpoint_name
    continuation = machine.continuation_event is not None
    expected_invocation = "continuation" if continuation else "development"
    if (
        receipt.invocation_kind != expected_invocation
        or checkpoint == "publication_recovery_pre"
        or not machine.runtime_receipts
        and (
            checkpoint != "attempt_open"
            or machine.intents
            or machine.responses
        )
    ):
        raise V315StorePoisoned("runtime_binding_receipt_invalid")

    if checkpoint == "attempt_open":
        if continuation:
            if (
                machine.continuation_event is None
                or event.previous_event_sha256
                != machine.continuation_event.event_sha256
                or not _fresh_process_boundary_safe(machine)
            ):
                raise V315StorePoisoned("runtime_binding_receipt_invalid")
        else:
            development_opens = tuple(
                item
                for item in machine.runtime_receipts
                if item.invocation_kind == "development"
                and item.checkpoint_name == "attempt_open"
            )
            if (
                not _fresh_process_boundary_safe(machine)
                or not development_opens
                and event.previous_event_sha256
                != machine.events[0].event_sha256
            ):
                raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    invocation_open = next(
        (
            item
            for item in reversed(machine.runtime_receipts)
            if item.invocation_kind == expected_invocation
            and item.checkpoint_name == "attempt_open"
        ),
        None,
    )
    if invocation_open is None:
        raise V315StorePoisoned("runtime_binding_receipt_invalid")
    if checkpoint.startswith("yahoo_"):
        expected = f"yahoo_{machine.yahoo_responses + 1:02d}_pre"
        if (
            continuation
            or checkpoint != expected
            or machine.active_intent is not None
            or machine.pending_subject_event_sha256 is not None
            or machine.open_segment_id is not None
            or machine.segments
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    if checkpoint == "model_initial_pre":
        if (
            continuation
            or machine.yahoo_responses != YAHOO_REQUEST_COUNT
            or machine.open_segment_id is not None
            or machine.segments
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    if checkpoint in {"model_initial_post", "model_continuation_post"}:
        index = 0 if checkpoint == "model_initial_post" else 1
        if (
            continuation != (index == 1)
            or len(machine.segments) != index + 1
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        close_checkpoint = machine.checkpoints.get(
            machine.segments[index].close_response_event_sha256
        )
        if (
            close_checkpoint is None
            or event.previous_event_sha256 != close_checkpoint.event_sha256
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    if checkpoint == "pause_publish_pre":
        if (
            continuation
            or len(machine.segments) != 1
            or machine.segments[0].generation_count != PILOT_COUNT
            or _model_post_binding_missing(machine)
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    if checkpoint == "model_continuation_pre":
        if (
            not continuation
            or not machine.pause_committed
            or len(machine.segments) != 1
            or machine.open_segment_id is not None
            or machine.runtime_receipts[-1] != invocation_open
            or event.previous_event_sha256 != invocation_open.event_sha256
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    if checkpoint == "evaluation_pre":
        if (
            not _private_values_ready(machine)
            or machine.market_values_opened
            or machine.model_responses_opened
            or _model_post_binding_missing(machine)
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    if checkpoint == "evaluation_post":
        if (
            not _private_values_ready(machine)
            or not machine.market_values_opened
            or not machine.model_responses_opened
            or _model_post_binding_missing(machine)
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    if checkpoint == "result_publish_pre":
        if (
            not _private_values_ready(machine)
            or not machine.market_values_opened
            or not machine.model_responses_opened
            or _model_post_binding_missing(machine)
        ):
            raise V315StorePoisoned("runtime_binding_receipt_invalid")
        return
    raise V315StorePoisoned("runtime_binding_receipt_invalid")


def _apply_response(machine: _Machine, event: JournalEvent) -> None:
    payload = event.payload
    if not _exact_keys(
        payload,
        {
            "request_intent_event_sha256",
            "request_id",
            "response_payload_sha256",
            "response_payload_bytes",
            "body_sha256",
            "body_bytes",
            "duration_ns",
        },
    ):
        raise V315StorePoisoned("response_event_invalid")
    active = machine.active_intent
    if (
        active is None
        or payload["request_intent_event_sha256"] != active.event_sha256
        or payload["request_id"] != active.request_id
        or not valid_sha256(payload["response_payload_sha256"])
        or type(payload["response_payload_bytes"]) is not int
        or payload["response_payload_bytes"] <= 0
        or not valid_sha256(payload["body_sha256"])
        or type(payload["body_bytes"]) is not int
        or payload["body_bytes"] < 0
    ):
        raise V315StorePoisoned("response_event_invalid")
    if active.effect_kind == "yahoo" and not (
        1 <= payload["body_bytes"] <= YAHOO_RESPONSE_CAP_BYTES
    ):
        raise V315StorePoisoned("yahoo_response_bytes_invalid")
    if active.effect_kind == "identity" and not (
        1 <= payload["body_bytes"] <= MODEL_RESPONSE_CAP_BYTES
    ):
        raise V315StorePoisoned("runtime_probe_response_bytes_invalid")
    if active.effect_kind == "gemma" and not (
        1 <= payload["body_bytes"] <= MODEL_RESPONSE_CAP_BYTES
    ):
        raise V315StorePoisoned("model_response_bytes_invalid")
    duration = payload["duration_ns"]
    if active.effect_kind == "gemma":
        if type(duration) is not int or duration <= 0:
            raise V315StorePoisoned("pilot_timing_invalid")
        machine.segment_durations.append(duration)
        machine.gemma_responses += 1
    elif duration is not None:
        raise V315StorePoisoned("response_duration_invalid")

    if active.effect_kind == "yahoo":
        machine.yahoo_responses += 1
        machine.yahoo_body_bytes += payload["body_bytes"]
        if machine.yahoo_body_bytes > YAHOO_BATCH_BODY_CAP_BYTES:
            raise V315StorePoisoned("yahoo_batch_body_cap_exceeded")
    elif active.effect_kind == "identity":
        machine.identity_responses += 1
        if active.model_phase == "pre_probe_start":
            machine.open_segment_id = active.segment_id
            machine.segment_phase = "pre"
            machine.segment_pre_count = 1
            machine.segment_post_count = 0
            machine.segment_durations = []
        elif active.model_phase == "pre_probe":
            machine.segment_pre_count = 2
            machine.segment_phase = "generation"
        elif active.model_phase == "post_probe":
            machine.segment_post_count = 1
            machine.segment_phase = "post"
        elif active.model_phase == "post_probe_close":
            machine.segment_post_count = 2
            summary = SegmentSummary(
                active.segment_id or "",
                len(machine.segment_durations),
                machine.segment_pre_count + machine.segment_post_count,
                tuple(machine.segment_durations),
                event.event_sha256,
            )
            machine.segments.append(summary)
            machine.open_segment_id = None
            machine.segment_phase = None
            machine.segment_pre_count = 0
            machine.segment_post_count = 0
            machine.segment_durations = []

    machine.responses[event.event_sha256] = event
    machine.active_intent = None
    machine.pending_subject_event_sha256 = event.event_sha256
    machine.pending_subject_type = "response"


def _replay_state(
    replay: JournalReplay, expected_authority: Mapping[str, Any]
) -> _Machine:
    machine = _new_machine(replay.authority_sha256, replay.events)
    if not replay.events:
        raise V315StorePoisoned("attempt_intent_missing")
    for index, event in enumerate(replay.events):
        # A terminal marker closes the hash chain at the state-machine layer.
        # Checking this before event-type or payload dispatch makes the rule
        # universal: even an otherwise unknown or malformed event cannot be
        # interpreted after a terminal outcome.
        if machine.terminal_event is not None:
            raise V315StorePoisoned("event_after_terminal")
        if event.event_type not in _EVENT_TYPES:
            raise V315StorePoisoned("foreign_event")
        payload = event.payload
        if (
            machine.runtime_binding_failed is not None
            and event.event_type not in {"candidate_abandoned", "attempt_terminal"}
        ):
            raise V315StorePoisoned("event_after_runtime_binding_failure")
        if event.event_type == "attempt_intent":
            if (
                index != 0
                or not _exact_keys(payload, {"authority"})
                or payload["authority"] != dict(expected_authority)
            ):
                raise V315StorePoisoned("attempt_intent_invalid")
            continue
        if index == 0:
            raise V315StorePoisoned("attempt_intent_missing")

        if event.event_type == "runtime_binding_receipt":
            receipt = _decode_runtime_binding_receipt(event)
            expected_previous = (
                machine.runtime_receipts[-1].runtime_binding_receipt_sha256
                if machine.runtime_receipts
                else None
            )
            if (
                receipt.invocation_kind == "publication_recovery"
                or receipt.previous_runtime_binding_receipt_sha256
                != expected_previous
                or not machine.runtime_receipts
                and receipt.checkpoint_name != "attempt_open"
            ):
                raise V315StorePoisoned("runtime_binding_receipt_invalid")
            _validate_attempt_receipt_transition(machine, event, receipt)
            machine.runtime_receipts.append(receipt)
            continue

        if event.event_type == "runtime_binding_failed":
            if not _exact_keys(
                payload,
                {
                    "checkpoint_name",
                    "failure_code",
                    "external_intent_count",
                    "previous_runtime_binding_receipt_sha256",
                    "journal_head_before_binding_failure_sha256",
                },
            ):
                raise V315StorePoisoned("runtime_binding_failure_invalid")
            expected_previous = (
                machine.runtime_receipts[-1].runtime_binding_receipt_sha256
                if machine.runtime_receipts
                else None
            )
            if (
                machine.runtime_binding_failed is not None
                or payload["checkpoint_name"] not in _RUNTIME_CHECKPOINTS
                or payload["checkpoint_name"] == "publication_recovery_pre"
                or payload["failure_code"] != "runtime_binding_mismatch"
                or type(payload["external_intent_count"]) is not int
                or payload["external_intent_count"] != len(machine.intents)
                or payload["previous_runtime_binding_receipt_sha256"]
                != expected_previous
                or payload["journal_head_before_binding_failure_sha256"]
                != event.previous_event_sha256
            ):
                raise V315StorePoisoned("runtime_binding_failure_invalid")
            machine.runtime_binding_failed = event
            continue

        if event.event_type == "candidate_abandoned":
            if not _exact_keys(
                payload,
                {
                    "candidate_sha256",
                    "original_proposed_status",
                    "effective_status",
                    "reason_code",
                    "publication_recovery_receipt_sha256",
                    "journal_head_before_abandon_sha256",
                },
            ):
                raise V315StorePoisoned("candidate_abandonment_invalid")
            if (
                machine.candidate_abandoned is not None
                or not valid_sha256(payload["candidate_sha256"])
                or payload["original_proposed_status"]
                not in {"completed", "rejected", "indeterminate"}
                or payload["effective_status"] != "indeterminate"
                or payload["reason_code"] != "unbound_terminal_candidate"
                or not valid_sha256(
                    payload["publication_recovery_receipt_sha256"]
                )
                or payload["journal_head_before_abandon_sha256"]
                != event.previous_event_sha256
            ):
                raise V315StorePoisoned("candidate_abandonment_invalid")
            machine.candidate_abandoned = event
            continue

        if event.event_type == "request_intent":
            if not _exact_keys(
                payload,
                {
                    "request_id",
                    "effect_kind",
                    "request_sha256",
                    "ordinal",
                    "segment_id",
                    "model_phase",
                },
            ):
                raise V315StorePoisoned("request_intent_invalid")
            intent = RequestIntent(
                event.event_sha256,
                payload["request_id"],
                payload["effect_kind"],
                payload["request_sha256"],
                payload["ordinal"],
                payload["segment_id"],
                payload["model_phase"],
            )
            if (
                not _safe_id(intent.request_id)
                or intent.effect_kind not in _EFFECT_KINDS
                or not valid_sha256(intent.request_sha256)
                or type(intent.ordinal) is not int
                or intent.ordinal <= 0
                or (
                    intent.segment_id is not None
                    and not _safe_id(intent.segment_id)
                )
                or intent.model_phase not in _MODEL_PHASES
            ):
                raise V315StorePoisoned("request_intent_invalid")
            _validate_request_transition(
                machine,
                intent,
                previous_event_sha256=event.previous_event_sha256,
            )
            if intent.model_phase == "pre_probe_start":
                # The segment is open from the durable pre-probe intent, not
                # merely after a response arrives.
                machine.open_segment_id = intent.segment_id
                machine.segment_phase = "pre"
                machine.segment_pre_count = 0
                machine.segment_post_count = 0
                machine.segment_durations = []
            machine.intents[event.event_sha256] = intent
            machine.intent_ids.add(intent.request_id)
            machine.active_intent = intent
            continue

        if event.event_type == "response_committed":
            if machine.pending_subject_event_sha256 is not None:
                raise V315StorePoisoned("response_order_invalid")
            _apply_response(machine, event)
            continue

        if event.event_type == "checkpoint_committed":
            if not _exact_keys(
                payload,
                {
                    "subject_event_sha256",
                    "subject_type",
                    "checkpoint_sha256",
                    "checkpoint_bytes",
                },
            ):
                raise V315StorePoisoned("checkpoint_event_invalid")
            if (
                payload["subject_event_sha256"]
                != machine.pending_subject_event_sha256
                or payload["subject_type"] != machine.pending_subject_type
                or payload["subject_type"] not in {"response", "pause"}
                or not valid_sha256(payload["checkpoint_sha256"])
                or type(payload["checkpoint_bytes"]) is not int
                or payload["checkpoint_bytes"] <= 0
            ):
                raise V315StorePoisoned("checkpoint_event_invalid")
            machine.checkpoints[payload["subject_event_sha256"]] = event
            if payload["subject_type"] == "pause":
                machine.pause_committed = True
            machine.pending_subject_event_sha256 = None
            machine.pending_subject_type = None
            continue

        if event.event_type == "paused_for_justification":
            if not _exact_keys(
                payload,
                {
                    "pilot_count",
                    "pilot_durations_ns",
                    "projected_ns",
                    "threshold_ns",
                    "formula",
                    "pause_candidate_sha256",
                    "pause_artifact_sha256",
                    "pause_artifact_bytes",
                    "publication_binding_receipt_sha256",
                    "journal_head_before_pause_sha256",
                },
            ):
                raise V315StorePoisoned("pause_event_invalid")
            expected_durations = (
                machine.segments[0].generation_durations_ns
                if len(machine.segments) == 1
                else ()
            )
            projected = _projected_ns(machine)
            if (
                machine.active_intent is not None
                or machine.pending_subject_event_sha256 is not None
                or machine.open_segment_id is not None
                or len(machine.segments) != 1
                or machine.segments[0].generation_count != PILOT_COUNT
                or machine.pause_event is not None
                or payload["pilot_count"] != PILOT_COUNT
                or payload["pilot_durations_ns"] != list(expected_durations)
                or payload["projected_ns"] != projected
                or type(projected) is not int
                or projected <= PAUSE_THRESHOLD_NS
                or payload["threshold_ns"] != PAUSE_THRESHOLD_NS
                or payload["formula"] != "sum(pilot_duration_ns)+70*max(pilot_duration_ns)"
                or not valid_sha256(payload["pause_candidate_sha256"])
                or not valid_sha256(payload["pause_artifact_sha256"])
                or type(payload["pause_artifact_bytes"]) is not int
                or payload["pause_artifact_bytes"] <= 0
                or not valid_sha256(
                    payload["publication_binding_receipt_sha256"]
                )
                or payload["journal_head_before_pause_sha256"]
                != event.previous_event_sha256
            ):
                raise V315StorePoisoned("pause_event_invalid")
            machine.pause_event = event
            machine.pending_subject_event_sha256 = event.event_sha256
            machine.pending_subject_type = "pause"
            continue

        if event.event_type == "continuation_authorized":
            if not _exact_keys(
                payload,
                {
                    "pause_event_sha256",
                    "continuation_sha256",
                    "permission_sha256",
                    "continuation_commit",
                },
            ):
                raise V315StorePoisoned("continuation_event_invalid")
            previous_continuation = machine.continuation_event
            if (
                not machine.pause_committed
                or machine.pause_event is None
                or payload["pause_event_sha256"]
                != machine.pause_event.event_sha256
                or not valid_sha256(payload["continuation_sha256"])
                or not valid_sha256(payload["permission_sha256"])
                or not _valid_commit(payload["continuation_commit"])
                or not _fresh_process_boundary_safe(machine)
                or previous_continuation is not None
                and (
                    payload["continuation_sha256"]
                    != previous_continuation.payload["continuation_sha256"]
                    or payload["continuation_commit"]
                    != previous_continuation.payload["continuation_commit"]
                )
            ):
                raise V315StorePoisoned("continuation_event_invalid")
            machine.continuation_event = event
            continue

        if event.event_type in {
            "market_values_opened",
            "model_responses_opened",
        }:
            if not _exact_keys(payload, set()):
                raise V315StorePoisoned("private_values_open_event_invalid")
            _validate_private_open_transition(machine, event.event_type)
            if event.event_type == "market_values_opened":
                machine.market_values_opened = True
            else:
                machine.model_responses_opened = True
            continue

        if event.event_type == "attempt_terminal":
            if not _exact_keys(
                payload,
                {
                    "status",
                    "terminal_code",
                    "candidate_sha256",
                    "candidate_bytes",
                    "candidate_journal_head_before_candidate_sha256",
                    "publication_binding_receipt_sha256",
                    "journal_head_before_terminal_sha256",
                },
            ):
                raise V315StorePoisoned("terminal_event_invalid")
            if (
                machine.terminal_event is not None
                or payload["status"]
                not in {"completed", "rejected", "indeterminate"}
                or not _safe_code(payload["terminal_code"])
                or not valid_sha256(payload["candidate_sha256"])
                or type(payload["candidate_bytes"]) is not int
                or payload["candidate_bytes"] <= 0
                or not valid_sha256(
                    payload["candidate_journal_head_before_candidate_sha256"]
                )
                or not valid_sha256(
                    payload["publication_binding_receipt_sha256"]
                )
                or payload["journal_head_before_terminal_sha256"]
                != event.previous_event_sha256
                or machine.pause_committed
                and machine.continuation_event is None
                or machine.active_intent is not None
                and payload["status"] != "indeterminate"
                or machine.candidate_abandoned is not None
                and (
                    payload["status"] != "indeterminate"
                    or payload["terminal_code"]
                    != "unbound_terminal_candidate"
                    or payload["candidate_sha256"]
                    != machine.candidate_abandoned.payload["candidate_sha256"]
                )
            ):
                raise V315StorePoisoned("terminal_event_invalid")
            if payload["status"] == "completed":
                expected_identity = 4 if len(machine.segments) == 1 else 8
                if (
                    machine.active_intent is not None
                    or machine.pending_subject_event_sha256 is not None
                    or machine.open_segment_id is not None
                    or machine.yahoo_responses != YAHOO_REQUEST_COUNT
                    or machine.gemma_responses != TOTAL_GENERATIONS
                    or machine.identity_responses != expected_identity
                    or len(machine.segments) not in {1, 2}
                    or not machine.market_values_opened
                    or not machine.model_responses_opened
                ):
                    raise V315StorePoisoned("completed_effect_budget_invalid")
            machine.terminal_event = event
            machine.terminal_code = payload["terminal_code"]
            continue

    return machine


def _decode_canonical_file(path: Path, *, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        details = regular_file(path, code=code)
        if details.st_size <= 0 or details.st_size > _MAX_CONTENT_BYTES:
            raise V315StorePoisoned(code)
        encoded = read_regular_bytes(path, code=code)
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except V315StoreError:
        raise
    except (V315JournalError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise V315StorePoisoned(code) from None
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise V315StorePoisoned(code)
    return value, encoded


def _scan_content(directory: Path, *, code: str) -> dict[str, tuple[dict[str, Any], bytes]]:
    try:
        secure_directory(directory)
        entries = tuple(directory.iterdir())
    except V315JournalError as exc:
        raise _translate_journal_error(exc) from None
    except OSError:
        raise V315StorePoisoned(code) from None
    result: dict[str, tuple[dict[str, Any], bytes]] = {}
    for path in entries:
        match = _CONTENT_NAME_RE.fullmatch(path.name)
        if match is None:
            raise V315StorePoisoned(code)
        value, encoded = _decode_canonical_file(path, code=code)
        digest = sha256_bytes(encoded)
        if digest != match.group(1) or digest in result:
            raise V315StorePoisoned(code)
        result[digest] = (value, encoded)
    return result


def _load_response(
    machine: _Machine,
    event: JournalEvent,
    payloads: Mapping[str, tuple[dict[str, Any], bytes]],
) -> CommittedResponse:
    event_payload = event.payload
    digest = event_payload["response_payload_sha256"]
    stored = payloads.get(digest)
    if stored is None:
        raise V315StorePoisoned("response_payload_missing")
    value, encoded = stored
    if (
        len(encoded) != event_payload["response_payload_bytes"]
        or set(value)
        != {
            "schema_version",
            "request_intent_event_sha256",
            "request_id",
            "body_base64",
            "body_bytes",
            "body_sha256",
            "metadata",
        }
        or value["schema_version"] != RESPONSE_PAYLOAD_SCHEMA_VERSION
        or value["request_intent_event_sha256"]
        != event_payload["request_intent_event_sha256"]
        or value["request_id"] != event_payload["request_id"]
        or type(value["body_base64"]) is not str
        or value["body_bytes"] != event_payload["body_bytes"]
        or value["body_sha256"] != event_payload["body_sha256"]
        or type(value["metadata"]) is not dict
    ):
        raise V315StorePoisoned("response_payload_invalid")
    try:
        body = base64.b64decode(value["body_base64"], validate=True)
    except (ValueError, TypeError):
        raise V315StorePoisoned("response_payload_invalid") from None
    if len(body) != value["body_bytes"] or sha256_bytes(body) != value["body_sha256"]:
        raise V315StorePoisoned("response_payload_invalid")
    intent = machine.intents.get(event_payload["request_intent_event_sha256"])
    if intent is None:
        raise V315StorePoisoned("response_payload_invalid")
    checkpoint = machine.checkpoints.get(event.event_sha256)
    return CommittedResponse(
        event.event_sha256,
        intent,
        digest,
        len(encoded),
        value["body_sha256"],
        value["body_bytes"],
        event_payload["duration_ns"],
        checkpoint.event_sha256 if checkpoint is not None else None,
        checkpoint.payload["checkpoint_sha256"]
        if checkpoint is not None
        else None,
        body,
        _freeze_json(value["metadata"]),
    )


def _audit_contents(
    machine: _Machine, payload_dir: Path, checkpoint_dir: Path
) -> tuple[
    dict[str, CommittedResponse],
    dict[str, tuple[dict[str, Any], bytes]],
    tuple[str, ...],
]:
    payloads = _scan_content(payload_dir, code="payload_directory_poisoned")
    checkpoints = _scan_content(
        checkpoint_dir, code="checkpoint_directory_poisoned"
    )
    loaded: dict[str, CommittedResponse] = {}
    referenced_payloads: set[str] = set()
    for response_event_sha, response_event in machine.responses.items():
        response = _load_response(machine, response_event, payloads)
        loaded[response_event_sha] = response
        referenced_payloads.add(response.payload_sha256)
    if set(payloads) != referenced_payloads:
        raise V315StorePoisoned("orphan_response_payload")

    referenced_checkpoints: set[str] = set()
    for subject, checkpoint_event in machine.checkpoints.items():
        digest = checkpoint_event.payload["checkpoint_sha256"]
        stored = checkpoints.get(digest)
        if stored is None:
            raise V315StorePoisoned("checkpoint_missing")
        value, encoded = stored
        if (
            len(encoded) != checkpoint_event.payload["checkpoint_bytes"]
            or set(value) != {"schema_version", "subject_event_sha256", "state"}
            or value["schema_version"] != CHECKPOINT_SCHEMA_VERSION
            or value["subject_event_sha256"] != subject
            or type(value["state"]) is not dict
            or value["state"]
            != _derive_checkpoint_state(
                machine,
                subject,
                checkpoint_event.payload["subject_type"],
            )
        ):
            raise V315StorePoisoned("checkpoint_invalid")
        referenced_checkpoints.add(digest)

    candidates = tuple(sorted(set(checkpoints) - referenced_checkpoints))
    if candidates:
        # Exactly one marker-last candidate is permitted only while the journal
        # has one authoritative response/pause awaiting its checkpoint marker.
        if machine.pending_subject_event_sha256 is None or len(candidates) != 1:
            raise V315StorePoisoned("orphan_checkpoint")
        candidate = checkpoints[candidates[0]][0]
        if (
            set(candidate) != {"schema_version", "subject_event_sha256", "state"}
            or candidate["schema_version"] != CHECKPOINT_SCHEMA_VERSION
            or candidate["subject_event_sha256"]
            != machine.pending_subject_event_sha256
            or type(candidate["state"]) is not dict
            or candidate["state"]
            != _derive_checkpoint_state(
                machine,
                machine.pending_subject_event_sha256,
                machine.pending_subject_type or "",
            )
        ):
            raise V315StorePoisoned("orphan_checkpoint")
    return loaded, checkpoints, candidates


def _bundle_files(directory: Path, expected: frozenset[str], *, code: str) -> None:
    try:
        _directory_details(directory, code=code)
        names = frozenset(path.name for path in directory.iterdir())
    except V315StoreError:
        raise
    except OSError:
        raise V315StorePoisoned(code) from None
    if names != expected:
        raise V315StorePoisoned(code)


def _audit_terminal_candidate(
    machine: _Machine, directory: Path
) -> tuple[TerminalCandidateReceipt, TerminalEvidenceReceipt | None]:
    _bundle_files(
        directory,
        frozenset(
            {
                TERMINAL_CANDIDATE_FILENAME,
                TERMINAL_PUBLIC_RESULT_FILENAME,
                TERMINAL_COMPARISON_BEFORE_FILENAME,
                TERMINAL_COMPARISON_AFTER_FILENAME,
            }
        ),
        code="terminal_candidate_directory_poisoned",
    )
    value, encoded = _decode_canonical_file(
        directory / TERMINAL_CANDIDATE_FILENAME,
        code="terminal_candidate_invalid",
    )
    public_result = read_regular_bytes(
        directory / TERMINAL_PUBLIC_RESULT_FILENAME,
        code="terminal_candidate_invalid",
    )
    comparison_before = read_regular_bytes(
        directory / TERMINAL_COMPARISON_BEFORE_FILENAME,
        code="terminal_candidate_invalid",
    )
    comparison_after = read_regular_bytes(
        directory / TERMINAL_COMPARISON_AFTER_FILENAME,
        code="terminal_candidate_invalid",
    )
    keys = {
        "schema_version",
        "authority_sha256",
        "stage",
        "attempt_id",
        "candidate_sequence",
        "journal_head_before_candidate_sha256",
        "proposed_status",
        "proposed_terminal_code",
        "evidence",
        "terminal_material_sha256",
        "public_result_sha256",
        "public_result_bytes",
        "comparison_before_sha256",
        "comparison_after_sha256",
        "comparison_after_bytes",
        "candidate_sha256",
    }
    terminal_material = {
        key: value[key]
        for key in (
            "schema_version",
            "authority_sha256",
            "stage",
            "attempt_id",
            "candidate_sequence",
            "journal_head_before_candidate_sha256",
            "proposed_status",
            "proposed_terminal_code",
            "evidence",
        )
    } if set(value) == keys else {}
    try:
        candidate_sha256 = _payload_self_hash(value, "candidate_sha256")
    except V315StorePoisoned:
        raise V315StorePoisoned("terminal_candidate_invalid") from None
    if (
        set(value) != keys
        or value["schema_version"] != TERMINAL_CANDIDATE_SCHEMA_VERSION
        or value["authority_sha256"] != machine.authority_sha256
        or value["stage"] != "development"
        or value["attempt_id"] != ATTEMPT_ID
        or value["candidate_sequence"] != 1
        or not valid_sha256(value["journal_head_before_candidate_sha256"])
        or value["proposed_status"]
        not in {"completed", "rejected", "indeterminate"}
        or not _safe_code(value["proposed_terminal_code"])
        or type(value["evidence"]) is not dict
        or value["terminal_material_sha256"]
        != canonical_sha256(terminal_material)
        or value["public_result_sha256"] != sha256_bytes(public_result)
        or value["public_result_bytes"] != len(public_result)
        or value["comparison_before_sha256"]
        != sha256_bytes(comparison_before)
        or value["comparison_after_sha256"] != sha256_bytes(comparison_after)
        or value["comparison_after_bytes"] != len(comparison_after)
        or len(public_result) <= 0
        or len(comparison_after) <= 0
    ):
        raise V315StorePoisoned("terminal_candidate_invalid")
    candidate = TerminalCandidateReceipt(
        candidate_sha256,
        len(encoded),
        value["journal_head_before_candidate_sha256"],
        value["proposed_status"],
        value["proposed_terminal_code"],
        value["terminal_material_sha256"],
        value["public_result_sha256"],
        value["public_result_bytes"],
        value["comparison_before_sha256"],
        value["comparison_after_sha256"],
        value["comparison_after_bytes"],
        _freeze_json(value["evidence"]),
        _freeze_json(value),
        public_result,
        comparison_before,
        comparison_after,
    )
    if machine.candidate_abandoned is not None:
        abandonment = machine.candidate_abandoned.payload
        if (
            abandonment["candidate_sha256"] != candidate.candidate_sha256
            or abandonment["original_proposed_status"]
            != candidate.proposed_status
        ):
            raise V315StorePoisoned("candidate_abandonment_invalid")
    event = machine.terminal_event
    if event is None:
        return candidate, None
    payload = event.payload
    if (
        payload["candidate_sha256"] != candidate.candidate_sha256
        or payload["candidate_bytes"] != candidate.candidate_bytes
        or payload["candidate_journal_head_before_candidate_sha256"]
        != candidate.journal_head_before_candidate_sha256
        or machine.candidate_abandoned is None
        and (
            payload["status"] != candidate.proposed_status
            or payload["terminal_code"] != candidate.proposed_terminal_code
        )
    ):
        raise V315StorePoisoned("terminal_candidate_binding_invalid")
    terminal = TerminalEvidenceReceipt(
        event.event_sha256,
        candidate.candidate_sha256,
        candidate.candidate_bytes,
        payload["journal_head_before_terminal_sha256"],
        payload["status"],
        payload["terminal_code"],
        candidate.terminal_material_sha256,
        candidate.public_result_sha256,
        candidate.public_result_bytes,
        candidate.comparison_before_sha256,
        candidate.comparison_after_sha256,
        candidate.comparison_after_bytes,
        candidate.evidence,
        candidate.public_result,
        candidate.comparison_before,
        candidate.comparison_after,
    )
    return candidate, terminal


def _audit_pause_candidate(
    machine: _Machine, directory: Path
) -> PauseCandidateReceipt:
    _bundle_files(
        directory,
        frozenset({PAUSE_CANDIDATE_FILENAME, PAUSE_ARTIFACT_FILENAME}),
        code="pause_candidate_directory_poisoned",
    )
    value, encoded = _decode_canonical_file(
        directory / PAUSE_CANDIDATE_FILENAME,
        code="pause_candidate_invalid",
    )
    artifact = read_regular_bytes(
        directory / PAUSE_ARTIFACT_FILENAME,
        code="pause_candidate_invalid",
    )
    if set(value) != {
        "schema_version",
        "authority_sha256",
        "attempt_id",
        "journal_head_before_candidate_sha256",
        "pause_artifact_sha256",
        "pause_artifact_bytes",
        "pause_candidate_sha256",
    }:
        raise V315StorePoisoned("pause_candidate_invalid")
    try:
        digest = _payload_self_hash(value, "pause_candidate_sha256")
    except V315StorePoisoned:
        raise V315StorePoisoned("pause_candidate_invalid") from None
    if (
        value["schema_version"] != PAUSE_CANDIDATE_SCHEMA_VERSION
        or value["authority_sha256"] != machine.authority_sha256
        or value["attempt_id"] != ATTEMPT_ID
        or not valid_sha256(value["journal_head_before_candidate_sha256"])
        or value["pause_artifact_sha256"] != sha256_bytes(artifact)
        or value["pause_artifact_bytes"] != len(artifact)
        or len(artifact) <= 0
    ):
        raise V315StorePoisoned("pause_candidate_invalid")
    receipt = PauseCandidateReceipt(
        digest,
        len(encoded),
        value["journal_head_before_candidate_sha256"],
        value["pause_artifact_sha256"],
        value["pause_artifact_bytes"],
        _freeze_json(value),
        artifact,
    )
    if machine.pause_event is not None:
        payload = machine.pause_event.payload
        if (
            payload["pause_candidate_sha256"] != digest
            or payload["pause_artifact_sha256"] != receipt.pause_artifact_sha256
            or payload["pause_artifact_bytes"] != receipt.pause_artifact_bytes
        ):
            raise V315StorePoisoned("pause_candidate_binding_invalid")
    return receipt


def _audit_publication_candidates(
    machine: _Machine, root: Path
) -> tuple[
    TerminalCandidateReceipt | None,
    TerminalEvidenceReceipt | None,
    PauseCandidateReceipt | None,
]:
    try:
        secure_directory(root)
        names = frozenset(path.name for path in root.iterdir())
    except (V315JournalError, OSError):
        raise V315StorePoisoned("publication_candidate_directory_poisoned") from None
    if not names.issubset({TERMINAL_BUNDLE_DIRECTORY, PAUSE_BUNDLE_DIRECTORY}):
        raise V315StorePoisoned("publication_candidate_directory_poisoned")
    terminal_candidate = None
    terminal = None
    pause_candidate = None
    if TERMINAL_BUNDLE_DIRECTORY in names:
        terminal_candidate, terminal = _audit_terminal_candidate(
            machine, root / TERMINAL_BUNDLE_DIRECTORY
        )
    elif machine.terminal_event is not None or machine.candidate_abandoned is not None:
        raise V315StorePoisoned("terminal_candidate_missing")
    if PAUSE_BUNDLE_DIRECTORY in names:
        pause_candidate = _audit_pause_candidate(
            machine, root / PAUSE_BUNDLE_DIRECTORY
        )
    elif machine.pause_event is not None:
        raise V315StorePoisoned("pause_candidate_missing")
    return terminal_candidate, terminal, pause_candidate


def _replay_recovery_state(
    replay: JournalReplay, attempt: _Machine
) -> _RecoveryMachine:
    machine = _RecoveryMachine(replay.events, [], [], [], "idle")
    current_intent: RecoveryIntentReceipt | None = None
    current_receipt: RuntimeBindingReceipt | None = None
    for event in replay.events:
        payload = event.payload
        if event.event_type not in _RECOVERY_EVENT_TYPES:
            raise V315StorePoisoned("recovery_foreign_event")
        if event.event_type == "recovery_intent":
            if (
                machine.terminals
                and machine.terminals[-1].outcome == "published"
                and not (
                    machine.terminals[-1].target_kind == "pause"
                    and payload.get("target_kind") != "pause"
                )
            ):
                raise V315StorePoisoned("recovery_after_publication")
            if not _exact_keys(
                payload,
                {
                    "ordinal",
                    "target_kind",
                    "candidate_sha256",
                    "attempt_terminal_event_sha256",
                    "public_artifact_sha256",
                    "public_artifact_bytes",
                    "comparison_before_sha256",
                    "comparison_after_sha256",
                    "observed_state_sha256",
                    "previous_recovery_terminal_sha256",
                    "recovery_intent_sha256",
                },
            ):
                raise V315StorePoisoned("recovery_intent_invalid")
            expected_ordinal = len(machine.intents) + 1
            expected_previous = _greatest_recovery_terminal_sha256(
                machine.terminals
            )
            if (
                type(payload["ordinal"]) is not int
                or payload["ordinal"] != expected_ordinal
                or payload["target_kind"] not in _RECOVERY_TARGET_KINDS
                or not valid_sha256(payload["candidate_sha256"])
                or (
                    payload["attempt_terminal_event_sha256"] is not None
                    and not valid_sha256(
                        payload["attempt_terminal_event_sha256"]
                    )
                )
                or any(
                    not valid_sha256(payload[key])
                    for key in (
                        "public_artifact_sha256",
                        "comparison_before_sha256",
                        "comparison_after_sha256",
                        "observed_state_sha256",
                    )
                )
                or type(payload["public_artifact_bytes"]) is not int
                or payload["public_artifact_bytes"] <= 0
                or payload["previous_recovery_terminal_sha256"]
                != expected_previous
                or payload["target_kind"] == "pause"
                and payload["attempt_terminal_event_sha256"] is not None
                or payload["target_kind"]
                in {"terminal_bound", "terminal_demoted_bound"}
                and payload["attempt_terminal_event_sha256"] is None
                or payload["target_kind"]
                not in {"pause", "terminal_bound", "terminal_demoted_bound"}
                and payload["attempt_terminal_event_sha256"] is not None
            ):
                raise V315StorePoisoned("recovery_intent_invalid")
            try:
                digest = _payload_self_hash(payload, "recovery_intent_sha256")
            except V315StorePoisoned:
                raise V315StorePoisoned("recovery_intent_invalid") from None
            current_intent = RecoveryIntentReceipt(
                event.event_sha256,
                payload["ordinal"],
                payload["target_kind"],
                digest,
                payload["candidate_sha256"],
                payload["attempt_terminal_event_sha256"],
                payload["previous_recovery_terminal_sha256"],
                _freeze_json(payload),
            )
            machine.intents.append(current_intent)
            current_receipt = None
            machine.phase = "intent"
            continue
        if event.event_type == "runtime_binding_receipt":
            if current_intent is None or machine.phase != "intent":
                raise V315StorePoisoned("recovery_receipt_order_invalid")
            receipt = _decode_runtime_binding_receipt(event)
            if machine.runtime_receipts:
                predecessor_valid = (
                    receipt.previous_runtime_binding_receipt_sha256
                    == machine.runtime_receipts[-1].runtime_binding_receipt_sha256
                )
            else:
                attempt_predecessors = {
                    item.runtime_binding_receipt_sha256
                    for item in attempt.runtime_receipts
                }
                predecessor_valid = (
                    receipt.previous_runtime_binding_receipt_sha256
                    in attempt_predecessors
                    if attempt_predecessors
                    else receipt.previous_runtime_binding_receipt_sha256 is None
                    and current_intent.target_kind
                    == "zero_effect_binding_rejection"
                )
            if (
                receipt.checkpoint_name != "publication_recovery_pre"
                or receipt.invocation_kind != "publication_recovery"
                or not predecessor_valid
            ):
                raise V315StorePoisoned("recovery_receipt_invalid")
            machine.runtime_receipts.append(receipt)
            current_receipt = receipt
            machine.phase = "receipt"
            continue
        if current_intent is None or current_receipt is None or machine.phase != "receipt":
            raise V315StorePoisoned("recovery_terminal_order_invalid")
        if not _exact_keys(
            payload,
            {
                "ordinal",
                "target_kind",
                "recovery_intent_sha256",
                "runtime_binding_receipt_sha256",
                "outcome",
                "final_state_sha256",
                "recovery_terminal_sha256",
            },
        ):
            raise V315StorePoisoned("recovery_terminal_invalid")
        if (
            payload["ordinal"] != current_intent.ordinal
            or payload["target_kind"] != current_intent.target_kind
            or payload["recovery_intent_sha256"]
            != current_intent.recovery_intent_sha256
            or payload["runtime_binding_receipt_sha256"]
            != current_receipt.runtime_binding_receipt_sha256
            or payload["outcome"] not in {"published", "rejected"}
            or not valid_sha256(payload["final_state_sha256"])
        ):
            raise V315StorePoisoned("recovery_terminal_invalid")
        try:
            digest = _payload_self_hash(payload, "recovery_terminal_sha256")
        except V315StorePoisoned:
            raise V315StorePoisoned("recovery_terminal_invalid") from None
        terminal = RecoveryTerminalReceipt(
            event.event_sha256,
            current_intent.ordinal,
            current_intent.target_kind,
            digest,
            payload["outcome"],
            payload["final_state_sha256"],
            _freeze_json(payload),
        )
        machine.terminals.append(terminal)
        machine.phase = "terminal"
    return machine


def _recovery_receipt_links(
    recovery: _RecoveryMachine,
) -> dict[str, RecoveryIntentReceipt]:
    intents_by_event = {item.event_sha256: item for item in recovery.intents}
    links: dict[str, RecoveryIntentReceipt] = {}
    for receipt in recovery.runtime_receipts:
        event = next(
            (
                item
                for item in recovery.events
                if item.event_sha256 == receipt.event_sha256
            ),
            None,
        )
        if event is None:
            raise V315StorePoisoned("recovery_receipt_invalid")
        intent = intents_by_event.get(event.previous_event_sha256)
        if intent is None:
            raise V315StorePoisoned("recovery_receipt_invalid")
        links[receipt.runtime_binding_receipt_sha256] = intent
    return links


def _audit_publication_links(
    attempt: _Machine,
    recovery: _RecoveryMachine,
    terminal_candidate: TerminalCandidateReceipt | None,
    pause_candidate: PauseCandidateReceipt | None,
) -> None:
    normal_receipts = {
        item.runtime_binding_receipt_sha256: item
        for item in attempt.runtime_receipts
    }
    normal_events = _attempt_receipt_events(attempt)
    recovery_links = _recovery_receipt_links(recovery)
    attempt_events = _attempt_events_by_sha(attempt)
    all_attempt_hashes = set(attempt_events)

    if terminal_candidate is not None:
        prefix_event = attempt_events.get(
            terminal_candidate.journal_head_before_candidate_sha256
        )
        normal_candidate = (
            prefix_event is not None
            and prefix_event.event_type == "runtime_binding_receipt"
            and prefix_event.payload["checkpoint_name"] == "evaluation_post"
        )
        created_matches = tuple(
            intent
            for intent in recovery.intents
            if intent.target_kind
            in {
                "terminal_indeterminate_created",
                "zero_effect_binding_rejection",
            }
            and intent.candidate_sha256 == terminal_candidate.candidate_sha256
            and _recovery_intent_values(intent)
            == (
                terminal_candidate.public_result_sha256,
                terminal_candidate.public_result_bytes,
                terminal_candidate.comparison_before_sha256,
                terminal_candidate.comparison_after_sha256,
            )
        )
        if (
            prefix_event is None
            or not normal_candidate
            and not created_matches
            or any(
                item.target_kind == "zero_effect_binding_rejection"
                and (
                    terminal_candidate.proposed_status != "rejected"
                    or terminal_candidate.proposed_terminal_code
                    != "runtime_binding_mismatch"
                )
                or item.target_kind == "terminal_indeterminate_created"
                and terminal_candidate.proposed_status != "indeterminate"
                for item in created_matches
            )
        ):
            raise V315StorePoisoned("terminal_candidate_predecessor_invalid")
    if pause_candidate is not None:
        prefix_event = attempt_events.get(
            pause_candidate.journal_head_before_candidate_sha256
        )
        if (
            prefix_event is None
            or prefix_event.event_type != "runtime_binding_receipt"
            or prefix_event.payload["checkpoint_name"]
            != "model_initial_post"
        ):
            raise V315StorePoisoned("pause_candidate_predecessor_invalid")

    if attempt.pause_event is not None:
        if pause_candidate is None:
            raise V315StorePoisoned("pause_candidate_missing")
        binding = attempt.pause_event.payload[
            "publication_binding_receipt_sha256"
        ]
        normal = normal_receipts.get(binding)
        normal_event = normal_events.get(binding)
        recovery_intent = recovery_links.get(binding)
        if normal is not None:
            if (
                normal.checkpoint_name != "pause_publish_pre"
                or normal_event is None
                or normal_event.previous_event_sha256
                != pause_candidate.journal_head_before_candidate_sha256
                or attempt.pause_event.previous_event_sha256
                != normal_event.event_sha256
            ):
                raise V315StorePoisoned("pause_publication_binding_invalid")
        elif (
            recovery_intent is None
            or recovery_intent.target_kind != "pause"
            or recovery_intent.candidate_sha256
            != pause_candidate.pause_candidate_sha256
            or _recovery_intent_values(recovery_intent)[:2]
            != (
                pause_candidate.pause_artifact_sha256,
                pause_candidate.pause_artifact_bytes,
            )
            or recovery_intent.payload["comparison_before_sha256"]
            != recovery_intent.payload["comparison_after_sha256"]
        ):
            raise V315StorePoisoned("pause_publication_binding_invalid")

    if attempt.candidate_abandoned is not None:
        if terminal_candidate is None:
            raise V315StorePoisoned("terminal_candidate_missing")
        payload = attempt.candidate_abandoned.payload
        intent = recovery_links.get(
            payload["publication_recovery_receipt_sha256"]
        )
        if (
            intent is None
            or intent.target_kind != "terminal_demoted"
            or intent.candidate_sha256 != terminal_candidate.candidate_sha256
            or attempt.candidate_abandoned.previous_event_sha256
            != attempt.candidate_abandoned.payload[
                "journal_head_before_abandon_sha256"
            ]
        ):
            raise V315StorePoisoned("candidate_abandonment_binding_invalid")

    if attempt.terminal_event is not None:
        if terminal_candidate is None:
            raise V315StorePoisoned("terminal_candidate_missing")
        binding = attempt.terminal_event.payload[
            "publication_binding_receipt_sha256"
        ]
        normal = normal_receipts.get(binding)
        normal_event = normal_events.get(binding)
        recovery_intent = recovery_links.get(binding)
        if normal is not None:
            if (
                normal.checkpoint_name != "result_publish_pre"
                or attempt.candidate_abandoned is not None
                or normal_event is None
                or normal_event.previous_event_sha256
                != terminal_candidate.journal_head_before_candidate_sha256
                or attempt.terminal_event.previous_event_sha256
                != normal_event.event_sha256
            ):
                raise V315StorePoisoned("terminal_publication_binding_invalid")
        elif recovery_intent is None:
            raise V315StorePoisoned("terminal_publication_binding_invalid")
        elif attempt.candidate_abandoned is not None:
            if (
                recovery_intent.target_kind != "terminal_demoted"
                or recovery_intent.candidate_sha256
                != terminal_candidate.candidate_sha256
                or attempt.terminal_event.previous_event_sha256
                != attempt.candidate_abandoned.event_sha256
            ):
                raise V315StorePoisoned("terminal_publication_binding_invalid")
        elif (
            recovery_intent.target_kind
            not in {
                "terminal_indeterminate_created",
                "zero_effect_binding_rejection",
            }
            or recovery_intent.candidate_sha256
            != terminal_candidate.candidate_sha256
            or _recovery_intent_values(recovery_intent)
            != (
                terminal_candidate.public_result_sha256,
                terminal_candidate.public_result_bytes,
                terminal_candidate.comparison_before_sha256,
                terminal_candidate.comparison_after_sha256,
            )
            or attempt.terminal_event.previous_event_sha256
            != terminal_candidate.journal_head_before_candidate_sha256
        ):
            raise V315StorePoisoned("terminal_publication_binding_invalid")

    first_demotion_values: tuple[str, int, str, str] | None = None
    for intent in recovery.intents:
        if intent.target_kind == "pause":
            if (
                pause_candidate is None
                or intent.candidate_sha256
                != pause_candidate.pause_candidate_sha256
                or _recovery_intent_values(intent)[:2]
                != (
                    pause_candidate.pause_artifact_sha256,
                    pause_candidate.pause_artifact_bytes,
                )
                or intent.payload["comparison_before_sha256"]
                != intent.payload["comparison_after_sha256"]
            ):
                raise V315StorePoisoned("recovery_candidate_binding_invalid")
        elif intent.target_kind == "terminal_bound":
            if (
                terminal_candidate is None
                or attempt.terminal_event is None
                or attempt.candidate_abandoned is not None
                or intent.candidate_sha256
                != terminal_candidate.candidate_sha256
                or intent.attempt_terminal_event_sha256
                != attempt.terminal_event.event_sha256
                or _recovery_intent_values(intent)
                != (
                    terminal_candidate.public_result_sha256,
                    terminal_candidate.public_result_bytes,
                    terminal_candidate.comparison_before_sha256,
                    terminal_candidate.comparison_after_sha256,
                )
            ):
                raise V315StorePoisoned("recovery_candidate_binding_invalid")
        elif intent.target_kind == "terminal_demoted":
            values = _recovery_intent_values(intent)
            if (
                terminal_candidate is None
                or intent.candidate_sha256
                != terminal_candidate.candidate_sha256
                or intent.attempt_terminal_event_sha256 is not None
                or intent.payload["comparison_before_sha256"]
                != terminal_candidate.comparison_before_sha256
                or first_demotion_values is not None
                and values != first_demotion_values
            ):
                raise V315StorePoisoned("recovery_candidate_binding_invalid")
            first_demotion_values = values
        elif intent.target_kind == "terminal_demoted_bound":
            linked = None
            if attempt.terminal_event is not None:
                linked = recovery_links.get(
                    attempt.terminal_event.payload[
                        "publication_binding_receipt_sha256"
                    ]
                )
            if (
                terminal_candidate is None
                or attempt.terminal_event is None
                or attempt.candidate_abandoned is None
                or intent.candidate_sha256
                != terminal_candidate.candidate_sha256
                or intent.attempt_terminal_event_sha256
                != attempt.terminal_event.event_sha256
                or linked is None
                or linked.target_kind != "terminal_demoted"
                or _recovery_intent_values(intent)
                != _recovery_intent_values(linked)
            ):
                raise V315StorePoisoned("recovery_candidate_binding_invalid")
        elif terminal_candidate is not None:
            if (
                intent.candidate_sha256
                != terminal_candidate.candidate_sha256
                or _recovery_intent_values(intent)
                != (
                    terminal_candidate.public_result_sha256,
                    terminal_candidate.public_result_bytes,
                    terminal_candidate.comparison_before_sha256,
                    terminal_candidate.comparison_after_sha256,
                )
            ):
                raise V315StorePoisoned("recovery_candidate_binding_invalid")

    intents_by_ordinal = {item.ordinal: item for item in recovery.intents}
    for terminal in recovery.terminals:
        if terminal.outcome != "published":
            continue
        intent = intents_by_ordinal[terminal.ordinal]
        if intent.target_kind == "pause":
            complete = attempt.pause_event is not None and attempt.pause_committed
        elif intent.target_kind == "terminal_demoted":
            complete = (
                attempt.candidate_abandoned is not None
                and attempt.terminal_event is not None
            )
        else:
            complete = (
                terminal_candidate is not None
                and attempt.terminal_event is not None
            )
        if not complete:
            raise V315StorePoisoned("recovery_publication_incomplete")


def _root_layout(root: Path) -> None:
    try:
        secure_directory(root)
        names = frozenset(path.name for path in root.iterdir())
    except V315JournalError as exc:
        raise _translate_journal_error(exc) from None
    except OSError:
        raise V315StorePoisoned("attempt_root_invalid") from None
    if names != _ROOT_ENTRIES:
        raise V315StorePoisoned("attempt_root_poisoned")


def _content_bytes(subject_event_sha256: str, state: Mapping[str, Any]) -> bytes:
    if not isinstance(state, Mapping):
        raise V315StoreError("checkpoint_state_invalid")
    value = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "subject_event_sha256": subject_event_sha256,
        "state": _thaw_json(state),
    }
    return canonical_json_bytes(value)


class AttemptStore:
    """Exclusively own and advance one immutable v3.15 attempt namespace."""

    def __init__(
        self,
        root: Path,
        authority: Mapping[str, Any],
        *,
        _create: bool = False,
    ) -> None:
        self.root = root
        self.authority = _validate_authority(authority)
        self.authority_sha256 = canonical_sha256(self.authority)
        self._closed = False
        self._poisoned = False
        self._lock: _AttemptLock | None = None
        self._journal: HashChainJournal | None = None
        self._recovery_journal: HashChainJournal | None = None
        self._machine: _Machine | None = None
        self._recovery_machine: _RecoveryMachine | None = None
        self._responses: dict[str, CommittedResponse] = {}
        self._checkpoints: dict[str, tuple[dict[str, Any], bytes]] = {}
        self._checkpoint_candidates: tuple[str, ...] = ()
        self._terminal_evidence: TerminalEvidenceReceipt | None = None
        self._terminal_candidate: TerminalCandidateReceipt | None = None
        self._pause_candidate: PauseCandidateReceipt | None = None
        self._live_intent_hashes: set[str] = set()
        self._live_model_segments: set[str] = set()
        self._live_pending_subjects: set[str] = set()
        self._live_attempt_runtime_receipts: set[str] = set()
        self._live_recovery_runtime_receipts: set[str] = set()
        self._live_terminal_candidate_sha256: str | None = None
        self._live_pause_candidate_sha256: str | None = None
        self._recovering_subject: str | None = None

        if not isinstance(root, Path) or not root.is_absolute():
            raise V315StoreError("attempt_root_invalid")
        lock_bytes = canonical_json_bytes(
            {
                "schema_version": LOCK_SCHEMA_VERSION,
                "authority_sha256": self.authority_sha256,
            }
        )
        try:
            if _create:
                _initialize_attempt_root(
                    root,
                    authority=self.authority,
                    authority_sha256=self.authority_sha256,
                    lock_bytes=lock_bytes,
                )
            _root_layout(root)
            self._lock = _AttemptLock(root / LOCK_FILENAME, lock_bytes)
            self._lock.acquire()
            self._journal = HashChainJournal(
                root / JOURNAL_DIRECTORY,
                authority_sha256=self.authority_sha256,
            )
            self._recovery_journal = HashChainJournal(
                root / RECOVERY_JOURNAL_DIRECTORY,
                authority_sha256=self.authority_sha256,
            )
            self._refresh()
        except Exception:
            self._close_lock()
            raise

    @classmethod
    def create(
        cls, root: Path, *, authority: Mapping[str, Any]
    ) -> "AttemptStore":
        return cls(root, authority, _create=True)

    @classmethod
    def open(
        cls, root: Path, *, authority: Mapping[str, Any]
    ) -> "AttemptStore":
        return cls(root, authority, _create=False)

    def __enter__(self) -> "AttemptStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _close_lock(self) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def close(self) -> None:
        if not self._closed:
            self._close_lock()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise V315StoreError("store_closed")
        if self._poisoned:
            raise V315StorePoisoned("store_poisoned")

    def privacy_scan_locked_payloads(self) -> Mapping[str, bytes]:
        """Return exact bytes for files that this process locks against reread."""

        self._require_open()
        if self._lock is None:
            raise V315StoreError("attempt_lock_invalid")
        return MappingProxyType({LOCK_FILENAME: bytes(self._lock.expected_bytes)})

    def _refresh(self) -> None:
        self._require_open()
        try:
            _root_layout(self.root)
            replay = replay_journal(
                self.root / JOURNAL_DIRECTORY,
                authority_sha256=self.authority_sha256,
            )
            machine = _replay_state(replay, self.authority)
            responses, checkpoints, candidates = _audit_contents(
                machine,
                self.root / PAYLOAD_DIRECTORY,
                self.root / CHECKPOINT_DIRECTORY,
            )
            recovery_replay = replay_journal(
                self.root / RECOVERY_JOURNAL_DIRECTORY,
                authority_sha256=self.authority_sha256,
            )
            recovery_machine = _replay_recovery_state(recovery_replay, machine)
            terminal_candidate, terminal_evidence, pause_candidate = (
                _audit_publication_candidates(
                    machine,
                    self.root / PUBLICATION_CANDIDATE_DIRECTORY,
                )
            )
            _audit_publication_links(
                machine,
                recovery_machine,
                terminal_candidate,
                pause_candidate,
            )
            self._machine = machine
            self._recovery_machine = recovery_machine
            self._responses = responses
            self._checkpoints = checkpoints
            self._checkpoint_candidates = candidates
            self._terminal_evidence = terminal_evidence
            self._terminal_candidate = terminal_candidate
            self._pause_candidate = pause_candidate
        except V315JournalError as exc:
            self._poisoned = True
            raise _translate_journal_error(exc) from None
        except V315StorePoisoned:
            self._poisoned = True
            raise

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> JournalEvent:
        self._require_open()
        if self._journal is None:
            raise V315StoreError("store_not_initialized")
        try:
            event = self._journal.append(event_type, payload)
            self._refresh()
            return event
        except V315JournalError as exc:
            self._poisoned = True
            raise _translate_journal_error(exc) from None

    def _append_recovery(
        self, event_type: str, payload: Mapping[str, Any]
    ) -> JournalEvent:
        self._require_open()
        if self._recovery_journal is None:
            raise V315StoreError("store_not_initialized")
        try:
            event = self._recovery_journal.append(event_type, payload)
            self._refresh()
            return event
        except V315JournalError as exc:
            self._poisoned = True
            raise _translate_journal_error(exc) from None

    def _machine_or_raise(self) -> _Machine:
        self._require_open()
        if self._machine is None:
            raise V315StoreError("store_not_initialized")
        return self._machine

    def _recovery_machine_or_raise(self) -> _RecoveryMachine:
        self._require_open()
        if self._recovery_machine is None:
            raise V315StoreError("store_not_initialized")
        return self._recovery_machine

    @property
    def runtime_binding_failure(self) -> JournalEvent | None:
        return self._machine_or_raise().runtime_binding_failed

    def runtime_binding_receipts(self) -> tuple[RuntimeBindingReceipt, ...]:
        """Return the replay-audited attempt receipts in durable order."""

        return tuple(self._machine_or_raise().runtime_receipts)

    @property
    def terminal_candidate(self) -> TerminalCandidateReceipt | None:
        self._require_open()
        return self._terminal_candidate

    @property
    def pause_candidate(self) -> PauseCandidateReceipt | None:
        self._require_open()
        return self._pause_candidate

    @property
    def terminal_candidate_abandoned(self) -> bool:
        """Expose only the replay-audited demotion marker's presence."""

        return self._machine_or_raise().candidate_abandoned is not None

    def recovery_intents(self) -> tuple[RecoveryIntentReceipt, ...]:
        """Return replay-audited recovery intents in durable ordinal order."""

        return tuple(self._recovery_machine_or_raise().intents)

    def recovery_terminals(self) -> tuple[RecoveryTerminalReceipt, ...]:
        """Return replay-audited recovery terminals in durable ordinal order."""

        return tuple(self._recovery_machine_or_raise().terminals)

    @property
    def latest_recovery_intent(self) -> RecoveryIntentReceipt | None:
        machine = self._recovery_machine_or_raise()
        return machine.intents[-1] if machine.intents else None

    @property
    def latest_recovery_terminal(self) -> RecoveryTerminalReceipt | None:
        machine = self._recovery_machine_or_raise()
        return machine.terminals[-1] if machine.terminals else None

    def _runtime_binding_payload(
        self,
        *,
        checkpoint_name: str,
        invocation_kind: str,
        head: str,
        tree: str,
        clean_state_sha256: str,
        repository_runtime_manifest_sha256: str,
        execution_dependency_manifest_sha256: str,
        loaded_code_manifest_sha256: str,
        runtime_binding_authority_sha256: str,
        route_argv_sha256: str,
        process_environment_sha256: str,
        interpreter_identity_sha256: str,
        previous_runtime_binding_receipt_sha256: str | None,
    ) -> dict[str, Any]:
        body = {
            "schema_version": RUNTIME_BINDING_RECEIPT_SCHEMA_VERSION,
            "checkpoint_name": checkpoint_name,
            "invocation_kind": invocation_kind,
            "head": head,
            "tree": tree,
            "clean_state_sha256": clean_state_sha256,
            "repository_runtime_manifest_sha256": (
                repository_runtime_manifest_sha256
            ),
            "execution_dependency_manifest_sha256": (
                execution_dependency_manifest_sha256
            ),
            "loaded_code_manifest_sha256": loaded_code_manifest_sha256,
            "runtime_binding_authority_sha256": (
                runtime_binding_authority_sha256
            ),
            "route_argv_sha256": route_argv_sha256,
            "process_environment_sha256": process_environment_sha256,
            "interpreter_identity_sha256": interpreter_identity_sha256,
            "previous_runtime_binding_receipt_sha256": (
                previous_runtime_binding_receipt_sha256
            ),
        }
        if (
            checkpoint_name not in _RUNTIME_CHECKPOINTS
            or invocation_kind not in _INVOCATION_KINDS
            or not _valid_commit(head)
            or not _valid_commit(tree)
            or any(
                not valid_sha256(body[key])
                for key in (
                    "clean_state_sha256",
                    "repository_runtime_manifest_sha256",
                    "execution_dependency_manifest_sha256",
                    "loaded_code_manifest_sha256",
                    "runtime_binding_authority_sha256",
                    "route_argv_sha256",
                    "process_environment_sha256",
                    "interpreter_identity_sha256",
                )
            )
        ):
            raise V315StoreError("runtime_binding_receipt_invalid")
        return {
            **body,
            "runtime_binding_receipt_sha256": canonical_sha256(body),
        }

    def append_runtime_binding_receipt(
        self,
        *,
        checkpoint_name: str,
        invocation_kind: str,
        head: str,
        tree: str,
        clean_state_sha256: str,
        repository_runtime_manifest_sha256: str,
        execution_dependency_manifest_sha256: str,
        loaded_code_manifest_sha256: str,
        runtime_binding_authority_sha256: str,
        route_argv_sha256: str,
        process_environment_sha256: str,
        interpreter_identity_sha256: str,
    ) -> RuntimeBindingReceipt:
        machine = self._machine_or_raise()
        if invocation_kind not in {"development", "continuation"}:
            raise V315StoreError("runtime_binding_receipt_invalid")
        previous = (
            machine.runtime_receipts[-1].runtime_binding_receipt_sha256
            if machine.runtime_receipts
            else None
        )
        payload = self._runtime_binding_payload(
            checkpoint_name=checkpoint_name,
            invocation_kind=invocation_kind,
            head=head,
            tree=tree,
            clean_state_sha256=clean_state_sha256,
            repository_runtime_manifest_sha256=repository_runtime_manifest_sha256,
            execution_dependency_manifest_sha256=execution_dependency_manifest_sha256,
            loaded_code_manifest_sha256=loaded_code_manifest_sha256,
            runtime_binding_authority_sha256=runtime_binding_authority_sha256,
            route_argv_sha256=route_argv_sha256,
            process_environment_sha256=process_environment_sha256,
            interpreter_identity_sha256=interpreter_identity_sha256,
            previous_runtime_binding_receipt_sha256=previous,
        )
        prospective = RuntimeBindingReceipt(
            "",
            checkpoint_name,
            invocation_kind,
            payload["runtime_binding_receipt_sha256"],
            previous,
            _freeze_json(payload),
        )
        synthetic_event = JournalEvent(
            len(machine.events) + 1,
            "runtime_binding_receipt",
            self.authority_sha256,
            machine.events[-1].event_sha256,
            MappingProxyType(payload),
            "",
            self.root / JOURNAL_DIRECTORY,
        )
        try:
            _validate_attempt_receipt_transition(
                machine, synthetic_event, prospective
            )
        except V315StorePoisoned:
            raise V315StoreError("runtime_binding_receipt_invalid") from None
        event = self._append("runtime_binding_receipt", payload)
        receipt = _decode_runtime_binding_receipt(event)
        self._live_attempt_runtime_receipts.add(
            receipt.runtime_binding_receipt_sha256
        )
        return receipt

    def record_runtime_binding_failure(
        self, *, checkpoint_name: str
    ) -> JournalEvent:
        machine = self._machine_or_raise()
        if (
            checkpoint_name not in _RUNTIME_CHECKPOINTS
            or checkpoint_name == "publication_recovery_pre"
            or machine.runtime_binding_failed is not None
            or machine.terminal_event is not None
        ):
            raise V315StoreError("runtime_binding_failure_invalid")
        previous = (
            machine.runtime_receipts[-1].runtime_binding_receipt_sha256
            if machine.runtime_receipts
            else None
        )
        return self._append(
            "runtime_binding_failed",
            {
                "checkpoint_name": checkpoint_name,
                "failure_code": "runtime_binding_mismatch",
                "external_intent_count": len(machine.intents),
                "previous_runtime_binding_receipt_sha256": previous,
                "journal_head_before_binding_failure_sha256": (
                    machine.events[-1].event_sha256
                ),
            },
        )

    def build_terminal_candidate(
        self,
        *,
        proposed_status: str,
        proposed_terminal_code: str,
        evidence: Mapping[str, Any],
        public_result: bytes,
        comparison_before: bytes,
        comparison_after: bytes,
    ) -> TerminalCandidateReceipt:
        machine = self._machine_or_raise()
        head_receipt = _attempt_receipt_at_head(machine)
        normal_evaluated = (
            head_receipt is not None
            and head_receipt.checkpoint_name == "evaluation_post"
            and machine.runtime_binding_failed is None
            and _private_values_ready(machine)
            and machine.market_values_opened
            and machine.model_responses_opened
        )
        exceptional = _exceptional_candidate_outcome(machine)
        if (
            self._terminal_candidate is not None
            or machine.terminal_event is not None
            or proposed_status not in {"completed", "rejected", "indeterminate"}
            or not _safe_code(proposed_terminal_code)
            or not isinstance(evidence, Mapping)
            or type(public_result) is not bytes
            or type(comparison_before) is not bytes
            or type(comparison_after) is not bytes
            or not public_result
            or not comparison_after
            or not normal_evaluated
            and (
                exceptional is None
                or proposed_status != exceptional[0]
                or proposed_terminal_code != exceptional[1]
            )
        ):
            raise V315StoreError("terminal_candidate_invalid")
        try:
            normalized_evidence = json.loads(
                canonical_json_bytes(_thaw_json(evidence))
            )
        except (V315JournalError, json.JSONDecodeError, TypeError, ValueError):
            raise V315StoreError("terminal_candidate_invalid") from None
        if type(normalized_evidence) is not dict:
            raise V315StoreError("terminal_candidate_invalid")
        prefix = machine.events[-1].event_sha256
        material = {
            "schema_version": TERMINAL_CANDIDATE_SCHEMA_VERSION,
            "authority_sha256": self.authority_sha256,
            "stage": "development",
            "attempt_id": ATTEMPT_ID,
            "candidate_sequence": 1,
            "journal_head_before_candidate_sha256": prefix,
            "proposed_status": proposed_status,
            "proposed_terminal_code": proposed_terminal_code,
            "evidence": normalized_evidence,
        }
        body = {
            **material,
            "terminal_material_sha256": canonical_sha256(material),
            "public_result_sha256": sha256_bytes(public_result),
            "public_result_bytes": len(public_result),
            "comparison_before_sha256": sha256_bytes(comparison_before),
            "comparison_after_sha256": sha256_bytes(comparison_after),
            "comparison_after_bytes": len(comparison_after),
        }
        candidate = {**body, "candidate_sha256": canonical_sha256(body)}
        encoded = canonical_json_bytes(candidate)
        return TerminalCandidateReceipt(
            candidate["candidate_sha256"],
            len(encoded),
            prefix,
            proposed_status,
            proposed_terminal_code,
            body["terminal_material_sha256"],
            body["public_result_sha256"],
            len(public_result),
            body["comparison_before_sha256"],
            body["comparison_after_sha256"],
            len(comparison_after),
            _freeze_json(normalized_evidence),
            _freeze_json(candidate),
            bytes(public_result),
            bytes(comparison_before),
            bytes(comparison_after),
        )

    def persist_terminal_candidate(
        self, candidate: TerminalCandidateReceipt
    ) -> TerminalCandidateReceipt:
        self._require_open()
        machine = self._machine_or_raise()
        head_receipt = _attempt_receipt_at_head(machine)
        normal_evaluated = (
            head_receipt is not None
            and head_receipt.checkpoint_name == "evaluation_post"
            and machine.runtime_binding_failed is None
            and _private_values_ready(machine)
            and machine.market_values_opened
            and machine.model_responses_opened
        )
        recovery = self._recovery_machine_or_raise()
        exceptional = _exceptional_candidate_outcome(machine)
        recovery_authorized = False
        if (
            exceptional is not None
            and recovery.phase == "receipt"
            and recovery.intents
            and recovery.runtime_receipts
        ):
            intent = recovery.intents[-1]
            recovery_authorized = (
                intent.target_kind == exceptional[2]
                and intent.candidate_sha256 == candidate.candidate_sha256
                and _recovery_intent_values(intent)
                == (
                    candidate.public_result_sha256,
                    candidate.public_result_bytes,
                    candidate.comparison_before_sha256,
                    candidate.comparison_after_sha256,
                )
                and candidate.proposed_status == exceptional[0]
                and candidate.proposed_terminal_code == exceptional[1]
            )
        if (
            not isinstance(candidate, TerminalCandidateReceipt)
            or self._terminal_candidate is not None
            or candidate.journal_head_before_candidate_sha256
            != machine.events[-1].event_sha256
            or not normal_evaluated
            and not recovery_authorized
        ):
            raise V315StoreError("terminal_candidate_invalid")
        directory = self.root / PUBLICATION_CANDIDATE_DIRECTORY / TERMINAL_BUNDLE_DIRECTORY
        _mkdir_new(directory)
        files = (
            (TERMINAL_PUBLIC_RESULT_FILENAME, candidate.public_result),
            (TERMINAL_COMPARISON_BEFORE_FILENAME, candidate.comparison_before),
            (TERMINAL_COMPARISON_AFTER_FILENAME, candidate.comparison_after),
            (
                TERMINAL_CANDIDATE_FILENAME,
                canonical_json_bytes(_thaw_json(candidate.candidate)),
            ),
        )
        for name, content in files:
            _write_exclusive(
                directory / name,
                content,
                code="terminal_candidate_write_failed",
            )
        fsync_directory(directory)
        self._refresh()
        if self._terminal_candidate is None:
            raise V315StorePoisoned("terminal_candidate_missing")
        self._live_terminal_candidate_sha256 = (
            self._terminal_candidate.candidate_sha256
        )
        return self._terminal_candidate

    def write_terminal_candidate(
        self,
        *,
        proposed_status: str,
        proposed_terminal_code: str,
        evidence: Mapping[str, Any],
        public_result: bytes,
        comparison_before: bytes,
        comparison_after: bytes,
    ) -> TerminalCandidateReceipt:
        return self.persist_terminal_candidate(
            self.build_terminal_candidate(
                proposed_status=proposed_status,
                proposed_terminal_code=proposed_terminal_code,
                evidence=evidence,
                public_result=public_result,
                comparison_before=comparison_before,
                comparison_after=comparison_after,
            )
        )

    def write_pause_candidate(
        self, *, pause_artifact: bytes
    ) -> PauseCandidateReceipt:
        machine = self._machine_or_raise()
        head_receipt = _attempt_receipt_at_head(machine)
        if (
            type(pause_artifact) is not bytes
            or not pause_artifact
            or self._pause_candidate is not None
            or machine.pause_event is not None
            or machine.active_intent is not None
            or machine.pending_subject_event_sha256 is not None
            or machine.open_segment_id is not None
            or len(machine.segments) != 1
            or machine.segments[0].generation_count != PILOT_COUNT
            or (_projected_ns(machine) or 0) <= PAUSE_THRESHOLD_NS
            or head_receipt is None
            or head_receipt.checkpoint_name != "model_initial_post"
            or head_receipt.invocation_kind != "development"
        ):
            raise V315StoreError("pause_candidate_invalid")
        body = {
            "schema_version": PAUSE_CANDIDATE_SCHEMA_VERSION,
            "authority_sha256": self.authority_sha256,
            "attempt_id": ATTEMPT_ID,
            "journal_head_before_candidate_sha256": (
                machine.events[-1].event_sha256
            ),
            "pause_artifact_sha256": sha256_bytes(pause_artifact),
            "pause_artifact_bytes": len(pause_artifact),
        }
        candidate = {
            **body,
            "pause_candidate_sha256": canonical_sha256(body),
        }
        directory = self.root / PUBLICATION_CANDIDATE_DIRECTORY / PAUSE_BUNDLE_DIRECTORY
        _mkdir_new(directory)
        _write_exclusive(
            directory / PAUSE_ARTIFACT_FILENAME,
            pause_artifact,
            code="pause_candidate_write_failed",
        )
        _write_exclusive(
            directory / PAUSE_CANDIDATE_FILENAME,
            canonical_json_bytes(candidate),
            code="pause_candidate_write_failed",
        )
        fsync_directory(directory)
        self._refresh()
        if self._pause_candidate is None:
            raise V315StorePoisoned("pause_candidate_missing")
        self._live_pause_candidate_sha256 = (
            self._pause_candidate.pause_candidate_sha256
        )
        return self._pause_candidate

    def begin_recovery(
        self,
        *,
        target_kind: str,
        candidate_sha256: str,
        attempt_terminal_event_sha256: str | None,
        public_artifact_sha256: str,
        public_artifact_bytes: int,
        comparison_before_sha256: str,
        comparison_after_sha256: str,
        observed_state_sha256: str,
    ) -> RecoveryIntentReceipt:
        recovery = self._recovery_machine_or_raise()
        attempt = self._machine_or_raise()
        if (
            recovery.terminals
            and recovery.terminals[-1].outcome == "published"
            and not (
                recovery.terminals[-1].target_kind == "pause"
                and target_kind != "pause"
            )
        ):
            raise V315StoreError("recovery_already_published")
        ordinal = len(recovery.intents) + 1
        previous = _greatest_recovery_terminal_sha256(recovery.terminals)
        if (
            target_kind not in _RECOVERY_TARGET_KINDS
            or not valid_sha256(candidate_sha256)
            or (
                attempt_terminal_event_sha256 is not None
                and not valid_sha256(attempt_terminal_event_sha256)
            )
            or any(
                not valid_sha256(item)
                for item in (
                    public_artifact_sha256,
                    comparison_before_sha256,
                    comparison_after_sha256,
                    observed_state_sha256,
                )
            )
            or type(public_artifact_bytes) is not int
            or public_artifact_bytes <= 0
        ):
            raise V315StoreError("recovery_intent_invalid")
        values = (
            public_artifact_sha256,
            public_artifact_bytes,
            comparison_before_sha256,
            comparison_after_sha256,
        )
        terminal_candidate = self._terminal_candidate
        pause_candidate = self._pause_candidate
        recovery_links = _recovery_receipt_links(recovery)
        if target_kind == "pause":
            if (
                pause_candidate is None
                or attempt.terminal_event is not None
                or candidate_sha256
                != pause_candidate.pause_candidate_sha256
                or attempt_terminal_event_sha256 is not None
                or public_artifact_sha256
                != pause_candidate.pause_artifact_sha256
                or public_artifact_bytes != pause_candidate.pause_artifact_bytes
                or comparison_before_sha256 != comparison_after_sha256
            ):
                raise V315StoreError("recovery_intent_invalid")
        elif target_kind == "terminal_bound":
            if (
                terminal_candidate is None
                or attempt.terminal_event is None
                or attempt.candidate_abandoned is not None
                or candidate_sha256 != terminal_candidate.candidate_sha256
                or attempt_terminal_event_sha256
                != attempt.terminal_event.event_sha256
                or values
                != (
                    terminal_candidate.public_result_sha256,
                    terminal_candidate.public_result_bytes,
                    terminal_candidate.comparison_before_sha256,
                    terminal_candidate.comparison_after_sha256,
                )
            ):
                raise V315StoreError("recovery_intent_invalid")
        elif target_kind == "terminal_demoted_bound":
            linked: RecoveryIntentReceipt | None = None
            if attempt.terminal_event is not None:
                linked = recovery_links.get(
                    attempt.terminal_event.payload[
                        "publication_binding_receipt_sha256"
                    ]
                )
            if (
                terminal_candidate is None
                or attempt.terminal_event is None
                or attempt.candidate_abandoned is None
                or candidate_sha256 != terminal_candidate.candidate_sha256
                or attempt_terminal_event_sha256
                != attempt.terminal_event.event_sha256
                or linked is None
                or linked.target_kind != "terminal_demoted"
                or linked.candidate_sha256 != candidate_sha256
                or values != _recovery_intent_values(linked)
            ):
                raise V315StoreError("recovery_intent_invalid")
        elif target_kind == "terminal_demoted":
            linked = None
            if attempt.candidate_abandoned is not None:
                linked = recovery_links.get(
                    attempt.candidate_abandoned.payload[
                        "publication_recovery_receipt_sha256"
                    ]
                )
            if (
                terminal_candidate is None
                or attempt.terminal_event is not None
                or candidate_sha256 != terminal_candidate.candidate_sha256
                or attempt_terminal_event_sha256 is not None
                or comparison_before_sha256
                != terminal_candidate.comparison_before_sha256
                or attempt.candidate_abandoned is not None
                and linked is None
                or linked is not None
                and (
                    linked.target_kind != "terminal_demoted"
                    or linked.candidate_sha256 != candidate_sha256
                    or values != _recovery_intent_values(linked)
                )
            ):
                raise V315StoreError("recovery_intent_invalid")
        else:
            exceptional = _exceptional_candidate_outcome(attempt)
            if (
                terminal_candidate is not None
                or attempt.terminal_event is not None
                or attempt_terminal_event_sha256 is not None
                or exceptional is None
                or target_kind != exceptional[2]
            ):
                raise V315StoreError("recovery_intent_invalid")
            prior = next(
                (
                    item
                    for item in reversed(recovery.intents)
                    if item.target_kind == target_kind
                ),
                None,
            )
            if prior is not None and (
                prior.candidate_sha256 != candidate_sha256
                or _recovery_intent_values(prior) != values
            ):
                raise V315StoreError("recovery_intent_invalid")
        body = {
            "ordinal": ordinal,
            "target_kind": target_kind,
            "candidate_sha256": candidate_sha256,
            "attempt_terminal_event_sha256": attempt_terminal_event_sha256,
            "public_artifact_sha256": public_artifact_sha256,
            "public_artifact_bytes": public_artifact_bytes,
            "comparison_before_sha256": comparison_before_sha256,
            "comparison_after_sha256": comparison_after_sha256,
            "observed_state_sha256": observed_state_sha256,
            "previous_recovery_terminal_sha256": previous,
        }
        payload = {**body, "recovery_intent_sha256": canonical_sha256(body)}
        event = self._append_recovery("recovery_intent", payload)
        refreshed = self._recovery_machine_or_raise().intents[-1]
        if refreshed.event_sha256 != event.event_sha256:
            raise V315StorePoisoned("recovery_intent_missing")
        return refreshed

    def append_recovery_runtime_binding_receipt(
        self,
        *,
        head: str,
        tree: str,
        clean_state_sha256: str,
        repository_runtime_manifest_sha256: str,
        execution_dependency_manifest_sha256: str,
        loaded_code_manifest_sha256: str,
        runtime_binding_authority_sha256: str,
        route_argv_sha256: str,
        process_environment_sha256: str,
        interpreter_identity_sha256: str,
    ) -> RuntimeBindingReceipt:
        recovery = self._recovery_machine_or_raise()
        attempt = self._machine_or_raise()
        if not recovery.intents or recovery.phase != "intent":
            raise V315StoreError("recovery_receipt_order_invalid")
        previous = (
            recovery.runtime_receipts[-1].runtime_binding_receipt_sha256
            if recovery.runtime_receipts
            else (
                attempt.runtime_receipts[-1].runtime_binding_receipt_sha256
                if attempt.runtime_receipts
                else None
            )
        )
        payload = self._runtime_binding_payload(
            checkpoint_name="publication_recovery_pre",
            invocation_kind="publication_recovery",
            head=head,
            tree=tree,
            clean_state_sha256=clean_state_sha256,
            repository_runtime_manifest_sha256=repository_runtime_manifest_sha256,
            execution_dependency_manifest_sha256=execution_dependency_manifest_sha256,
            loaded_code_manifest_sha256=loaded_code_manifest_sha256,
            runtime_binding_authority_sha256=runtime_binding_authority_sha256,
            route_argv_sha256=route_argv_sha256,
            process_environment_sha256=process_environment_sha256,
            interpreter_identity_sha256=interpreter_identity_sha256,
            previous_runtime_binding_receipt_sha256=previous,
        )
        event = self._append_recovery("runtime_binding_receipt", payload)
        receipt = _decode_runtime_binding_receipt(event)
        self._live_recovery_runtime_receipts.add(
            receipt.runtime_binding_receipt_sha256
        )
        return receipt

    def finish_recovery(
        self, *, outcome: str, final_state_sha256: str
    ) -> RecoveryTerminalReceipt:
        recovery = self._recovery_machine_or_raise()
        attempt = self._machine_or_raise()
        if (
            not recovery.intents
            or not recovery.runtime_receipts
            or recovery.phase != "receipt"
            or outcome not in {"published", "rejected"}
            or not valid_sha256(final_state_sha256)
            or recovery.runtime_receipts[-1].runtime_binding_receipt_sha256
            not in self._live_recovery_runtime_receipts
        ):
            raise V315StoreError("recovery_terminal_invalid")
        intent = recovery.intents[-1]
        receipt = recovery.runtime_receipts[-1]
        if outcome == "published":
            terminal = attempt.terminal_event
            if intent.target_kind == "pause":
                complete = (
                    attempt.pause_event is not None
                    and attempt.pause_committed
                    and self._pause_candidate is not None
                )
            elif intent.target_kind == "terminal_demoted":
                complete = (
                    terminal is not None
                    and attempt.candidate_abandoned is not None
                    and terminal.payload["publication_binding_receipt_sha256"]
                    == receipt.runtime_binding_receipt_sha256
                )
            elif intent.target_kind in {
                "terminal_indeterminate_created",
                "zero_effect_binding_rejection",
            }:
                complete = (
                    terminal is not None
                    and self._terminal_candidate is not None
                    and terminal.payload["publication_binding_receipt_sha256"]
                    == receipt.runtime_binding_receipt_sha256
                )
            else:
                complete = terminal is not None and self._terminal_candidate is not None
            if not complete:
                raise V315StoreError("recovery_terminal_invalid")
        body = {
            "ordinal": intent.ordinal,
            "target_kind": intent.target_kind,
            "recovery_intent_sha256": intent.recovery_intent_sha256,
            "runtime_binding_receipt_sha256": (
                receipt.runtime_binding_receipt_sha256
            ),
            "outcome": outcome,
            "final_state_sha256": final_state_sha256,
        }
        payload = {
            **body,
            "recovery_terminal_sha256": canonical_sha256(body),
        }
        event = self._append_recovery("recovery_terminal", payload)
        terminal = self._recovery_machine_or_raise().terminals[-1]
        if terminal.event_sha256 != event.event_sha256:
            raise V315StorePoisoned("recovery_terminal_missing")
        return terminal

    def abandon_terminal_candidate(
        self, *, publication_recovery_receipt_sha256: str
    ) -> JournalEvent:
        machine = self._machine_or_raise()
        candidate = self._terminal_candidate
        recovery = self._recovery_machine_or_raise()
        linked = _recovery_receipt_links(recovery).get(
            publication_recovery_receipt_sha256
        )
        if (
            candidate is None
            or machine.terminal_event is not None
            or machine.candidate_abandoned is not None
            or not valid_sha256(publication_recovery_receipt_sha256)
            or recovery.phase != "receipt"
            or not recovery.runtime_receipts
            or publication_recovery_receipt_sha256
            != recovery.runtime_receipts[-1].runtime_binding_receipt_sha256
            or publication_recovery_receipt_sha256
            not in self._live_recovery_runtime_receipts
            or linked is None
            or linked.target_kind != "terminal_demoted"
            or linked.candidate_sha256 != candidate.candidate_sha256
        ):
            raise V315StoreError("candidate_abandonment_invalid")
        return self._append(
            "candidate_abandoned",
            {
                "candidate_sha256": candidate.candidate_sha256,
                "original_proposed_status": candidate.proposed_status,
                "effective_status": "indeterminate",
                "reason_code": "unbound_terminal_candidate",
                "publication_recovery_receipt_sha256": (
                    publication_recovery_receipt_sha256
                ),
                "journal_head_before_abandon_sha256": (
                    machine.events[-1].event_sha256
                ),
            },
        )

    def append_attempt_terminal(
        self,
        *,
        publication_binding_receipt_sha256: str,
        status: str | None = None,
        terminal_code: str | None = None,
    ) -> TerminalEvidenceReceipt:
        machine = self._machine_or_raise()
        candidate = self._terminal_candidate
        recovery = self._recovery_machine_or_raise()
        attempt_receipt_event = _attempt_receipt_events(machine).get(
            publication_binding_receipt_sha256
        )
        recovery_intent = _recovery_receipt_links(recovery).get(
            publication_binding_receipt_sha256
        )
        if (
            candidate is None
            or machine.terminal_event is not None
            or not valid_sha256(publication_binding_receipt_sha256)
        ):
            raise V315StoreError("terminal_invalid")
        if machine.candidate_abandoned is not None:
            expected_status = "indeterminate"
            expected_code = "unbound_terminal_candidate"
        else:
            expected_status = candidate.proposed_status
            expected_code = candidate.proposed_terminal_code
        status = expected_status if status is None else status
        terminal_code = expected_code if terminal_code is None else terminal_code
        if status != expected_status or terminal_code != expected_code:
            raise V315StoreError("terminal_invalid")
        prefix = machine.events[-1].event_sha256
        if machine.candidate_abandoned is not None:
            authorized = (
                prefix == machine.candidate_abandoned.event_sha256
                and publication_binding_receipt_sha256
                in self._live_recovery_runtime_receipts
                and recovery_intent is not None
                and recovery_intent.target_kind == "terminal_demoted"
                and recovery_intent.candidate_sha256
                == candidate.candidate_sha256
            )
        elif attempt_receipt_event is not None:
            receipt = next(
                item
                for item in machine.runtime_receipts
                if item.runtime_binding_receipt_sha256
                == publication_binding_receipt_sha256
            )
            authorized = (
                receipt.checkpoint_name == "result_publish_pre"
                and publication_binding_receipt_sha256
                in self._live_attempt_runtime_receipts
                and self._live_terminal_candidate_sha256
                == candidate.candidate_sha256
                and prefix == attempt_receipt_event.event_sha256
                and attempt_receipt_event.previous_event_sha256
                == candidate.journal_head_before_candidate_sha256
            )
        else:
            authorized = (
                publication_binding_receipt_sha256
                in self._live_recovery_runtime_receipts
                and self._live_terminal_candidate_sha256
                == candidate.candidate_sha256
                and recovery_intent is not None
                and recovery_intent.target_kind
                in {
                    "terminal_indeterminate_created",
                    "zero_effect_binding_rejection",
                }
                and recovery_intent.candidate_sha256
                == candidate.candidate_sha256
                and _recovery_intent_values(recovery_intent)
                == (
                    candidate.public_result_sha256,
                    candidate.public_result_bytes,
                    candidate.comparison_before_sha256,
                    candidate.comparison_after_sha256,
                )
                and prefix
                == candidate.journal_head_before_candidate_sha256
            )
        if not authorized:
            raise V315StoreError("terminal_invalid")
        self._append(
            "attempt_terminal",
            {
                "status": status,
                "terminal_code": terminal_code,
                "candidate_sha256": candidate.candidate_sha256,
                "candidate_bytes": candidate.candidate_bytes,
                "candidate_journal_head_before_candidate_sha256": (
                    candidate.journal_head_before_candidate_sha256
                ),
                "publication_binding_receipt_sha256": (
                    publication_binding_receipt_sha256
                ),
                "journal_head_before_terminal_sha256": prefix,
            },
        )
        if self._terminal_evidence is None:
            raise V315StorePoisoned("terminal_evidence_missing")
        return self._terminal_evidence

    @property
    def snapshot(self) -> AttemptSnapshot:
        machine = self._machine_or_raise()
        status = "active"
        terminal_code = machine.terminal_code
        recoverable = False
        if machine.terminal_event is not None:
            status = str(machine.terminal_event.payload["status"])
        elif machine.runtime_binding_failed is not None:
            count = machine.runtime_binding_failed.payload[
                "external_intent_count"
            ]
            status = "rejected" if count == 0 else "indeterminate"
            terminal_code = (
                "runtime_binding_mismatch"
                if count == 0
                else _candidate_less_terminal_code(machine)
                or "post_intent_runtime_binding_failed"
            )
        elif machine.active_intent is not None:
            live = machine.active_intent.event_sha256 in self._live_intent_hashes
            status = "active" if live else "indeterminate"
            terminal_code = None if live else "request_outcome_unknown"
        elif machine.open_segment_id is not None:
            live = machine.open_segment_id in self._live_model_segments
            status = "active" if live else "indeterminate"
            terminal_code = (
                None if live else "irreversible_state_ambiguous"
            )
        elif machine.pending_subject_event_sha256 is not None:
            # Only a Yahoo response, a closed segment post-probe response, or a
            # local pause marker can be completed after a restart.
            if machine.pending_subject_type == "pause":
                recoverable = True
            else:
                response = self._responses[machine.pending_subject_event_sha256]
                recoverable = (
                    response.intent.effect_kind == "yahoo"
                    or response.intent.model_phase == "post_probe_close"
                )
            status = "checkpoint_recovery" if recoverable else "indeterminate"
            if not recoverable:
                terminal_code = "irreversible_state_ambiguous"
        elif _model_post_binding_missing(machine):
            status = "indeterminate"
            terminal_code = "model_post_binding_missing"
        elif machine.pause_committed and machine.continuation_event is None:
            status = "paused"

        projection = _projected_ns(machine)
        pause_required = (
            machine.open_segment_id is not None
            and not machine.segments
            and len(machine.segment_durations) >= PILOT_COUNT
            and projection is not None
            and projection > PAUSE_THRESHOLD_NS
        ) or (
            len(machine.segments) == 1
            and machine.segments[0].generation_count == PILOT_COUNT
            and not machine.pause_committed
        )
        return AttemptSnapshot(
            self.authority_sha256,
            machine.events[-1].event_sha256,
            len(machine.events),
            status,
            terminal_code,
            len(machine.intents),
            sum(
                intent.effect_kind == "yahoo"
                for intent in machine.intents.values()
            ),
            sum(
                intent.effect_kind == "identity"
                for intent in machine.intents.values()
            ),
            sum(
                intent.effect_kind == "gemma"
                for intent in machine.intents.values()
            ),
            len(machine.responses),
            len(machine.checkpoints),
            machine.yahoo_responses,
            machine.yahoo_body_bytes,
            machine.identity_responses,
            machine.gemma_responses,
            machine.active_intent.request_id
            if machine.active_intent is not None
            else None,
            machine.pending_subject_event_sha256,
            machine.open_segment_id,
            tuple(machine.segments),
            pause_required,
            machine.pause_committed,
            machine.continuation_event is not None,
            (
                machine.continuation_event.payload["continuation_sha256"]
                if machine.continuation_event is not None
                else None
            ),
            (
                machine.continuation_event.payload["permission_sha256"]
                if machine.continuation_event is not None
                else None
            ),
            (
                machine.continuation_event.payload["continuation_commit"]
                if machine.continuation_event is not None
                else None
            ),
            machine.market_values_opened,
            machine.model_responses_opened,
            recoverable,
        )

    def begin_request(
        self,
        *,
        request_id: str,
        effect_kind: str,
        request_sha256: str,
        segment_id: str | None = None,
        model_phase: str = "none",
    ) -> RequestIntent:
        machine = self._machine_or_raise()
        if self.snapshot.status in {
            "indeterminate",
            "checkpoint_recovery",
            "paused",
            "completed",
            "rejected",
        }:
            raise V315StoreError("request_not_authorized")
        payload = {
            "request_id": request_id,
            "effect_kind": effect_kind,
            "request_sha256": request_sha256,
            "ordinal": len(machine.intents) + 1,
            "segment_id": segment_id,
            "model_phase": model_phase,
        }
        # Validate before persistence using the same replay transition.
        candidate = RequestIntent(
            "f" * 64,
            request_id,
            effect_kind,
            request_sha256,
            payload["ordinal"],
            segment_id,
            model_phase,
        )
        if (
            not _safe_id(request_id)
            or effect_kind not in _EFFECT_KINDS
            or not valid_sha256(request_sha256)
            or (segment_id is not None and not _safe_id(segment_id))
            or model_phase not in _MODEL_PHASES
        ):
            raise V315StoreError("request_intent_invalid")
        _validate_request_transition(
            machine,
            candidate,
            previous_event_sha256=machine.events[-1].event_sha256,
        )
        event = self._append("request_intent", payload)
        self._live_intent_hashes.add(event.event_sha256)
        if model_phase == "pre_probe_start" and segment_id is not None:
            self._live_model_segments.add(segment_id)
        return self._machine_or_raise().intents[event.event_sha256]

    def commit_response(
        self,
        *,
        request_intent_event_sha256: str,
        body: bytes,
        metadata: Mapping[str, Any],
        duration_ns: int | None = None,
    ) -> CommittedResponse:
        machine = self._machine_or_raise()
        active = machine.active_intent
        if (
            active is None
            or active.event_sha256 != request_intent_event_sha256
            or active.event_sha256 not in self._live_intent_hashes
            or type(body) is not bytes
            or not isinstance(metadata, Mapping)
            or (active.effect_kind == "gemma" and (type(duration_ns) is not int or duration_ns <= 0))
            or (active.effect_kind != "gemma" and duration_ns is not None)
        ):
            raise V315StoreError("response_commit_invalid")
        if active.effect_kind == "yahoo" and not (
            1 <= len(body) <= YAHOO_RESPONSE_CAP_BYTES
        ):
            raise V315StoreError("yahoo_response_bytes_invalid")
        if active.effect_kind == "identity" and not (
            1 <= len(body) <= MODEL_RESPONSE_CAP_BYTES
        ):
            raise V315StoreError("runtime_probe_response_bytes_invalid")
        if active.effect_kind == "gemma" and not (
            1 <= len(body) <= MODEL_RESPONSE_CAP_BYTES
        ):
            raise V315StoreError("model_response_bytes_invalid")
        payload_value = {
            "schema_version": RESPONSE_PAYLOAD_SCHEMA_VERSION,
            "request_intent_event_sha256": active.event_sha256,
            "request_id": active.request_id,
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_bytes": len(body),
            "body_sha256": sha256_bytes(body),
            "metadata": _thaw_json(metadata),
        }
        encoded = canonical_json_bytes(payload_value)
        digest = sha256_bytes(encoded)
        self._write_content(
            self.root / PAYLOAD_DIRECTORY,
            digest,
            encoded,
            allow_existing_exact=False,
            conflict_code="response_payload_exists",
        )
        try:
            event = self._append(
                "response_committed",
                {
                    "request_intent_event_sha256": active.event_sha256,
                    "request_id": active.request_id,
                    "response_payload_sha256": digest,
                    "response_payload_bytes": len(encoded),
                    "body_sha256": payload_value["body_sha256"],
                    "body_bytes": len(body),
                    "duration_ns": duration_ns,
                },
            )
        except Exception:
            self._poisoned = True
            raise
        self._live_intent_hashes.discard(active.event_sha256)
        self._live_pending_subjects.add(event.event_sha256)
        return self._responses[event.event_sha256]

    def _write_content(
        self,
        directory: Path,
        digest: str,
        encoded: bytes,
        *,
        allow_existing_exact: bool,
        conflict_code: str,
    ) -> Path:
        self._require_open()
        if type(encoded) is not bytes or not 1 <= len(encoded) <= _MAX_CONTENT_BYTES:
            raise V315StoreError("content_size_invalid")
        if sha256_bytes(encoded) != digest:
            raise V315StoreError("content_hash_invalid")
        final = directory / f"{digest}.json"
        pending = directory / f".pending-{digest}.json"
        try:
            final_exists = entry_exists_nofollow(
                final, code="content_write_failed"
            )
            pending_exists = entry_exists_nofollow(
                pending, code="content_write_failed"
            )
        except V315JournalError as exc:
            self._poisoned = True
            raise V315StorePoisoned(exc.code) from None
        if final_exists:
            try:
                regular_file(final, code="content_write_failed")
            except V315JournalError as exc:
                self._poisoned = True
                raise V315StorePoisoned(exc.code) from None
            if allow_existing_exact:
                try:
                    if (
                        read_regular_bytes(
                            final, code="content_write_failed"
                        )
                        == encoded
                    ):
                        return final
                except V315JournalError as exc:
                    self._poisoned = True
                    raise V315StorePoisoned(exc.code) from None
            raise V315StoreConflict(conflict_code)
        if pending_exists:
            raise V315StorePoisoned("content_write_incomplete")
        try:
            _write_exclusive(pending, encoded, code="content_write_failed")
            try:
                publish_noreplace(
                    pending, final, code="content_destination_exists"
                )
            except V315JournalError as exc:
                try:
                    destination_exists = entry_exists_nofollow(
                        final, code="content_write_failed"
                    )
                except V315JournalError as presence_exc:
                    raise V315StorePoisoned(presence_exc.code) from None
                if destination_exists:
                    raise V315StoreConflict(conflict_code) from None
                raise V315StorePoisoned(exc.code) from None
            fsync_directory(directory)
            _value, reread = _decode_canonical_file(
                final, code="content_write_failed"
            )
            if reread != encoded:
                raise V315StorePoisoned("content_write_failed")
            return final
        except Exception:
            self._poisoned = True
            raise

    def commit_checkpoint(
        self,
        *,
        subject_event_sha256: str,
        state: Mapping[str, Any],
    ) -> JournalEvent:
        machine = self._machine_or_raise()
        if (
            machine.pending_subject_event_sha256 != subject_event_sha256
            or machine.pending_subject_type not in {"response", "pause"}
            or (
                subject_event_sha256 not in self._live_pending_subjects
                and subject_event_sha256 != self._recovering_subject
            )
        ):
            raise V315StoreError("checkpoint_subject_invalid")
        expected_state = _derive_checkpoint_state(
            machine,
            subject_event_sha256,
            machine.pending_subject_type,
        )
        if not isinstance(state, Mapping) or dict(state) != expected_state:
            raise V315StoreError("checkpoint_state_invalid")
        encoded = _content_bytes(subject_event_sha256, expected_state)
        digest = sha256_bytes(encoded)
        self._write_content(
            self.root / CHECKPOINT_DIRECTORY,
            digest,
            encoded,
            allow_existing_exact=True,
            conflict_code="checkpoint_conflict",
        )
        try:
            event = self._append(
                "checkpoint_committed",
                {
                    "subject_event_sha256": subject_event_sha256,
                    "subject_type": machine.pending_subject_type,
                    "checkpoint_sha256": digest,
                    "checkpoint_bytes": len(encoded),
                },
            )
            self._live_pending_subjects.discard(subject_event_sha256)
            return event
        except Exception:
            self._poisoned = True
            raise

    def commit_response_and_checkpoint(
        self,
        *,
        request_intent_event_sha256: str,
        body: bytes,
        metadata: Mapping[str, Any],
        checkpoint_state: Mapping[str, Any],
        duration_ns: int | None = None,
    ) -> tuple[CommittedResponse, JournalEvent]:
        response = self.commit_response(
            request_intent_event_sha256=request_intent_event_sha256,
            body=body,
            metadata=metadata,
            duration_ns=duration_ns,
        )
        checkpoint = self.commit_checkpoint(
            subject_event_sha256=response.event_sha256,
            state=checkpoint_state,
        )
        return response, checkpoint

    def derive_pending_checkpoint_state(self) -> Mapping[str, Any]:
        """Return the exact store-derived state for the one pending marker.

        Callers may use this for a live marker or deterministic crash recovery,
        but they cannot choose any checkpoint field: ``commit_checkpoint`` and
        replay both independently derive and compare the same value.
        """

        machine = self._machine_or_raise()
        subject = machine.pending_subject_event_sha256
        subject_type = machine.pending_subject_type
        if subject is None or subject_type not in {"response", "pause"}:
            raise V315StoreError("checkpoint_not_pending")
        return _derive_checkpoint_state(machine, subject, subject_type)

    def pending_checkpoint_context(self) -> PendingCheckpointContext:
        machine = self._machine_or_raise()
        subject = machine.pending_subject_event_sha256
        if subject is None:
            raise V315StoreError("checkpoint_not_pending")
        committed = self._responses.get(subject)
        response = (
            PendingResponseReceipt(
                committed.event_sha256,
                committed.intent,
                committed.payload_sha256,
                committed.payload_bytes,
                committed.body_sha256,
                committed.body_bytes,
                committed.duration_ns,
            )
            if committed is not None
            else None
        )
        pause_payload = (
            machine.pause_event.payload
            if machine.pending_subject_type == "pause"
            and machine.pause_event is not None
            else None
        )
        return PendingCheckpointContext(
            subject,
            machine.pending_subject_type or "",
            response,
            pause_payload,
            machine.events,
        )

    def committed_responses(
        self,
        *,
        effect_kind: str | None = None,
        segment_id: str | None = None,
    ) -> tuple[CommittedResponse, ...]:
        """Return an immutable, journal-ordered response batch after its seal.

        This is the sole public body-read path.  Yahoo stays closed until all
        six responses have marker-last checkpoints.  Runtime identity can be
        read only as a complete two-request pre-probe or four-request closed
        segment guard.  Gemma stays closed until the complete 75-generation
        guarded batch is closed; in particular, a five-row planned pause never
        releases pilot bodies.
        """

        machine = self._machine_or_raise()
        if effect_kind is not None and effect_kind not in _EFFECT_KINDS:
            raise V315StoreError("response_filter_invalid")
        if segment_id is not None and not _safe_id(segment_id):
            raise V315StoreError("response_filter_invalid")
        if effect_kind == "yahoo" and segment_id is not None:
            raise V315StoreError("response_filter_invalid")

        ordered = tuple(
            self._responses[event.event_sha256]
            for event in machine.events
            if event.event_type == "response_committed"
        )
        selected = tuple(
            response
            for response in ordered
            if (effect_kind is None or response.intent.effect_kind == effect_kind)
            and (
                segment_id is None
                or response.intent.segment_id == segment_id
            )
        )
        if any(
            response.event_sha256 not in machine.checkpoints
            for response in selected
        ):
            raise V315StoreError("response_batch_unsealed")

        if effect_kind is None:
            expected_identity = 4 if len(machine.segments) == 1 else 8
            if (
                segment_id is not None
                or machine.active_intent is not None
                or machine.pending_subject_event_sha256 is not None
                or machine.open_segment_id is not None
                or machine.yahoo_responses != YAHOO_REQUEST_COUNT
                or machine.gemma_responses != TOTAL_GENERATIONS
                or len(machine.segments) not in {1, 2}
                or machine.identity_responses != expected_identity
            ):
                raise V315StoreError("response_batch_unsealed")
            if not (
                machine.market_values_opened
                and machine.model_responses_opened
            ):
                raise V315StoreError("response_values_not_opened")
            return selected

        if effect_kind == "yahoo":
            yahoo = tuple(
                response
                for response in ordered
                if response.intent.effect_kind == "yahoo"
            )
            if (
                machine.yahoo_responses != YAHOO_REQUEST_COUNT
                or len(yahoo) != YAHOO_REQUEST_COUNT
                or any(
                    response.event_sha256 not in machine.checkpoints
                    for response in yahoo
                )
            ):
                raise V315StoreError("response_batch_unsealed")
            if not machine.market_values_opened:
                raise V315StoreError("response_values_not_opened")
            return yahoo

        if effect_kind == "gemma":
            # A paused five-row segment is guarded but deliberately opaque.  It
            # becomes readable only together with the closed remaining-70
            # segment, so callers cannot inspect pilot semantics while asking
            # the user for continuation permission.
            if (
                machine.gemma_responses != TOTAL_GENERATIONS
                or machine.open_segment_id is not None
                or machine.pending_subject_event_sha256 is not None
                or len(machine.segments) not in {1, 2}
                or [item.generation_count for item in machine.segments]
                not in ([TOTAL_GENERATIONS], [PILOT_COUNT, REMAINING_COUNT])
            ):
                raise V315StoreError("response_batch_unsealed")
            segment_ids = {item.segment_id for item in machine.segments}
            if segment_id is not None and segment_id not in segment_ids:
                raise V315StoreError("response_filter_invalid")
            for summary in machine.segments:
                if summary.close_response_event_sha256 not in machine.checkpoints:
                    raise V315StoreError("response_batch_unsealed")
            gemma = tuple(
                response
                for response in ordered
                if response.intent.effect_kind == "gemma"
                and (
                    segment_id is None
                    or response.intent.segment_id == segment_id
                )
            )
            expected = (
                TOTAL_GENERATIONS
                if segment_id is None
                else next(
                    item.generation_count
                    for item in machine.segments
                    if item.segment_id == segment_id
                )
            )
            if len(gemma) != expected or any(
                response.event_sha256 not in machine.checkpoints
                for response in gemma
            ):
                raise V315StoreError("response_batch_unsealed")
            if not machine.model_responses_opened:
                raise V315StoreError("response_values_not_opened")
            return gemma

        # Identity is operational evidence and must be opened to establish the
        # pre/post runtime guard.  It is nevertheless released only at a whole
        # probe boundary, never after just version or the first post request.
        identities = tuple(
            response
            for response in ordered
            if response.intent.effect_kind == "identity"
            and (
                segment_id is None
                or response.intent.segment_id == segment_id
            )
        )
        groups: dict[str, list[CommittedResponse]] = {}
        for response in identities:
            identity_segment = response.intent.segment_id
            if identity_segment is None:
                raise V315StorePoisoned("identity_segment_missing")
            groups.setdefault(identity_segment, []).append(response)
        if segment_id is not None and segment_id not in groups:
            raise V315StoreError("response_filter_invalid")
        closed_ids = {item.segment_id for item in machine.segments}
        for identity_segment, responses in groups.items():
            if identity_segment in closed_ids:
                expected = 4
            elif (
                identity_segment == machine.open_segment_id
                and machine.segment_phase == "generation"
                and machine.segment_pre_count == 2
                and machine.segment_post_count == 0
            ):
                expected = 2
            else:
                raise V315StoreError("response_batch_unsealed")
            if len(responses) != expected or any(
                response.event_sha256 not in machine.checkpoints
                for response in responses
            ):
                raise V315StoreError("response_batch_unsealed")
        return identities

    def committed_response_receipts(
        self,
        *,
        effect_kind: str,
        segment_id: str | None = None,
    ) -> tuple[CommittedResponseReceipt, ...]:
        """Return body-opaque, journal-ordered receipts at a sealed boundary.

        Unlike :meth:`committed_responses`, this permits the closed five-pilot
        segment.  That is necessary to rebuild its guard and timing-only pause
        after the post-probe is sealed, without exposing any model text.
        """

        machine = self._machine_or_raise()
        if effect_kind not in _EFFECT_KINDS:
            raise V315StoreError("response_filter_invalid")
        if segment_id is not None and not _safe_id(segment_id):
            raise V315StoreError("response_filter_invalid")
        if effect_kind == "yahoo" and segment_id is not None:
            raise V315StoreError("response_filter_invalid")
        ordered = tuple(
            self._responses[event.event_sha256]
            for event in machine.events
            if event.event_type == "response_committed"
        )
        selected = tuple(
            response
            for response in ordered
            if response.intent.effect_kind == effect_kind
            and (
                segment_id is None
                or response.intent.segment_id == segment_id
            )
        )
        if not selected or any(
            response.checkpoint_event_sha256 is None
            or response.checkpoint_sha256 is None
            for response in selected
        ):
            raise V315StoreError("response_batch_unsealed")

        if effect_kind == "yahoo":
            if machine.yahoo_responses != YAHOO_REQUEST_COUNT or len(selected) != YAHOO_REQUEST_COUNT:
                raise V315StoreError("response_batch_unsealed")
        elif effect_kind == "identity":
            # Reuse the body-read guard, which already proves a complete pre or
            # post probe.  The returned receipt remains body-opaque.
            self.committed_responses(
                effect_kind="identity", segment_id=segment_id
            )
        else:
            summaries = {
                item.segment_id: item for item in machine.segments
            }
            target_ids = (
                {segment_id}
                if segment_id is not None
                else {response.intent.segment_id for response in selected}
            )
            if None in target_ids or not target_ids:
                raise V315StoreError("response_filter_invalid")
            for target_id in target_ids:
                summary = summaries.get(target_id)
                if (
                    summary is None
                    or summary.close_response_event_sha256
                    not in machine.checkpoints
                ):
                    raise V315StoreError("response_batch_unsealed")
                count = sum(
                    response.intent.segment_id == target_id
                    for response in selected
                )
                if count != summary.generation_count:
                    raise V315StoreError("response_batch_unsealed")

        return tuple(
            CommittedResponseReceipt(
                response.event_sha256,
                response.intent,
                response.payload_sha256,
                response.payload_bytes,
                response.body_sha256,
                response.body_bytes,
                response.duration_ns,
                response.checkpoint_event_sha256,
                response.checkpoint_sha256,
            )
            for response in selected
            if response.checkpoint_event_sha256 is not None
            and response.checkpoint_sha256 is not None
        )

    def recover_pending_checkpoint(
        self,
        builder: Callable[[PendingCheckpointContext], Mapping[str, Any]],
    ) -> JournalEvent:
        if not callable(builder):
            raise V315StoreError("checkpoint_builder_invalid")
        snapshot = self.snapshot
        if snapshot.status != "checkpoint_recovery":
            raise V315StoreError("checkpoint_recovery_forbidden")
        context = self.pending_checkpoint_context()
        try:
            rebuilt = builder(context)
        except V315StoreError:
            raise
        except Exception:
            raise V315StoreError("checkpoint_rebuild_failed") from None
        if not isinstance(rebuilt, Mapping):
            raise V315StoreError("checkpoint_rebuild_failed")
        expected = dict(self.derive_pending_checkpoint_state())
        if dict(rebuilt) != expected:
            raise V315StoreError("checkpoint_rebuild_failed")
        encoded = _content_bytes(context.subject_event_sha256, expected)
        digest = sha256_bytes(encoded)
        if self._checkpoint_candidates:
            if self._checkpoint_candidates != (digest,):
                self._poisoned = True
                raise V315StorePoisoned("checkpoint_mismatch")
            stored = self._checkpoints[digest][1]
            if stored != encoded:
                self._poisoned = True
                raise V315StorePoisoned("checkpoint_mismatch")
        self._recovering_subject = context.subject_event_sha256
        try:
            return self.commit_checkpoint(
                subject_event_sha256=context.subject_event_sha256,
                state=expected,
            )
        finally:
            self._recovering_subject = None

    def commit_planned_pause(
        self,
        *,
        publication_binding_receipt_sha256: str,
        pause_candidate_sha256: str | None = None,
        checkpoint_state: Mapping[str, Any] | None = None,
    ) -> tuple[JournalEvent, JournalEvent]:
        machine = self._machine_or_raise()
        recovery = self._recovery_machine_or_raise()
        attempt_receipt_event = _attempt_receipt_events(machine).get(
            publication_binding_receipt_sha256
        )
        recovery_intent = _recovery_receipt_links(recovery).get(
            publication_binding_receipt_sha256
        )
        if (
            machine.active_intent is not None
            or machine.pending_subject_event_sha256 is not None
            or machine.open_segment_id is not None
            or len(machine.segments) != 1
            or machine.segments[0].generation_count != PILOT_COUNT
            or machine.pause_event is not None
            or self._pause_candidate is None
            or not valid_sha256(publication_binding_receipt_sha256)
        ):
            raise V315StoreError("pause_not_authorized")
        durations = machine.segments[0].generation_durations_ns
        projected = sum(durations) + REMAINING_COUNT * max(durations)
        if projected <= PAUSE_THRESHOLD_NS:
            raise V315StoreError("pause_threshold_not_crossed")
        candidate = self._pause_candidate
        if candidate is None or (
            pause_candidate_sha256 is not None
            and pause_candidate_sha256 != candidate.pause_candidate_sha256
        ):
            raise V315StoreError("pause_candidate_invalid")
        if attempt_receipt_event is not None:
            receipt = next(
                item
                for item in machine.runtime_receipts
                if item.runtime_binding_receipt_sha256
                == publication_binding_receipt_sha256
            )
            binding_authorized = (
                receipt.checkpoint_name == "pause_publish_pre"
                and publication_binding_receipt_sha256
                in self._live_attempt_runtime_receipts
                and self._live_pause_candidate_sha256
                == candidate.pause_candidate_sha256
                and machine.events[-1].event_sha256
                == attempt_receipt_event.event_sha256
                and attempt_receipt_event.previous_event_sha256
                == candidate.journal_head_before_candidate_sha256
            )
        else:
            current_head = machine.events[-1]
            old_normal_receipt = (
                current_head.event_type == "runtime_binding_receipt"
                and current_head.payload["checkpoint_name"]
                == "pause_publish_pre"
                and current_head.previous_event_sha256
                == candidate.journal_head_before_candidate_sha256
            )
            binding_authorized = (
                publication_binding_receipt_sha256
                in self._live_recovery_runtime_receipts
                and recovery_intent is not None
                and recovery_intent.target_kind == "pause"
                and recovery_intent.candidate_sha256
                == candidate.pause_candidate_sha256
                and (
                    current_head.event_sha256
                    == candidate.journal_head_before_candidate_sha256
                    or old_normal_receipt
                )
            )
        if not binding_authorized:
            raise V315StoreError("pause_not_authorized")

        pause_payload_base = {
            "pilot_count": PILOT_COUNT,
            "pilot_durations_ns": list(durations),
            "projected_ns": projected,
            "threshold_ns": PAUSE_THRESHOLD_NS,
            "formula": "sum(pilot_duration_ns)+70*max(pilot_duration_ns)",
            "pause_candidate_sha256": candidate.pause_candidate_sha256,
            "pause_artifact_sha256": candidate.pause_artifact_sha256,
            "pause_artifact_bytes": candidate.pause_artifact_bytes,
            "publication_binding_receipt_sha256": (
                publication_binding_receipt_sha256
            ),
            "journal_head_before_pause_sha256": machine.events[-1].event_sha256,
        }
        try:
            pause = self._append(
                "paused_for_justification",
                pause_payload_base,
            )
        except Exception:
            self._poisoned = True
            raise
        self._live_pending_subjects.add(pause.event_sha256)
        derived = dict(self.derive_pending_checkpoint_state())
        if checkpoint_state is not None and dict(checkpoint_state) != derived:
            raise V315StoreError("checkpoint_state_invalid")
        return pause, self.commit_checkpoint(
            subject_event_sha256=pause.event_sha256,
            state=derived,
        )

    def authorize_continuation(
        self,
        *,
        continuation_sha256: str,
        permission_sha256: str,
        continuation_commit: str,
    ) -> JournalEvent:
        machine = self._machine_or_raise()
        previous_continuation = machine.continuation_event
        if (
            not machine.pause_committed
            or machine.pause_event is None
            or not valid_sha256(continuation_sha256)
            or not valid_sha256(permission_sha256)
            or not _valid_commit(continuation_commit)
            or not _fresh_process_boundary_safe(machine)
            or previous_continuation is not None
            and (
                continuation_sha256
                != previous_continuation.payload["continuation_sha256"]
                or continuation_commit
                != previous_continuation.payload["continuation_commit"]
            )
        ):
            raise V315StoreError("continuation_not_authorized")
        return self._append(
            "continuation_authorized",
            {
                "pause_event_sha256": machine.pause_event.event_sha256,
                "continuation_sha256": continuation_sha256,
                "permission_sha256": permission_sha256,
                "continuation_commit": continuation_commit,
            },
        )

    def mark_market_values_opened(self) -> JournalEvent:
        """Durably record the conservative boundary before market parsing."""

        machine = self._machine_or_raise()
        try:
            _validate_private_open_transition(machine, "market_values_opened")
        except V315StorePoisoned:
            raise V315StoreError("private_values_open_not_authorized") from None
        return self._append("market_values_opened", {})

    def mark_model_responses_opened(self) -> JournalEvent:
        """Durably record the conservative boundary before model parsing."""

        machine = self._machine_or_raise()
        try:
            _validate_private_open_transition(machine, "model_responses_opened")
        except V315StorePoisoned:
            raise V315StoreError("private_values_open_not_authorized") from None
        return self._append("model_responses_opened", {})

    def committed_terminal_evidence(self) -> TerminalEvidenceReceipt:
        """Return the replay-audited private terminal evidence, if committed."""

        self._require_open()
        if self._terminal_evidence is None:
            raise V315StoreError("terminal_evidence_not_committed")
        return self._terminal_evidence


def audit_attempt(
    root: Path, *, authority: Mapping[str, Any]
) -> AttemptSnapshot:
    """Acquire the same exclusive lock, replay once, and return safe state."""

    with AttemptStore.open(root, authority=authority) as store:
        return store.snapshot


__all__ = [
    "ATTEMPT_ID",
    "AttemptSnapshot",
    "AttemptStore",
    "CHECKPOINT_SCHEMA_VERSION",
    "CommittedResponse",
    "CommittedResponseReceipt",
    "PAUSE_THRESHOLD_NS",
    "PAUSE_CANDIDATE_SCHEMA_VERSION",
    "PAUSE_CHECKPOINT_STATE_SCHEMA_VERSION",
    "PauseCandidateReceipt",
    "PendingResponseReceipt",
    "PILOT_COUNT",
    "PendingCheckpointContext",
    "REMAINING_COUNT",
    "RECOVERY_INTENT_SCHEMA_VERSION",
    "RECOVERY_TERMINAL_SCHEMA_VERSION",
    "RESPONSE_PAYLOAD_SCHEMA_VERSION",
    "RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION",
    "RecoveryIntentReceipt",
    "RecoveryTerminalReceipt",
    "RequestIntent",
    "RUNTIME_BINDING_RECEIPT_SCHEMA_VERSION",
    "RuntimeBindingReceipt",
    "SegmentSummary",
    "TERMINAL_CANDIDATE_SCHEMA_VERSION",
    "TERMINAL_EVIDENCE_SCHEMA_VERSION",
    "TerminalCandidateReceipt",
    "TerminalEvidenceReceipt",
    "TOTAL_GENERATIONS",
    "V315StoreConflict",
    "V315StoreError",
    "V315StorePoisoned",
    "YAHOO_REQUEST_COUNT",
    "YAHOO_BATCH_BODY_CAP_BYTES",
    "audit_attempt",
]
