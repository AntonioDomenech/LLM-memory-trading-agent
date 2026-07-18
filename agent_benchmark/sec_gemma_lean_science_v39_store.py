"""One-shot private attempt store for the v3.9 development experiment.

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

from agent_benchmark.sec_gemma_lean_science_v39_journal import (
    HashChainJournal,
    JournalEvent,
    JournalReplay,
    V39JournalError,
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
    "aapl-sec-gemma-lean-science-v3-9-attempt-lock-v1"
)
RESPONSE_PAYLOAD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-response-payload-v1"
)
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-checkpoint-v1"
)
RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-checkpoint-state-v1"
)
PAUSE_CHECKPOINT_STATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-pause-checkpoint-state-v2"
)
TERMINAL_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-terminal-evidence-v1"
)

ATTEMPT_ID: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-development-001"
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
TERMINAL_DIRECTORY: Final[str] = "terminal"
_ROOT_ENTRIES: Final[frozenset[str]] = frozenset(
    {
        LOCK_FILENAME,
        JOURNAL_DIRECTORY,
        PAYLOAD_DIRECTORY,
        CHECKPOINT_DIRECTORY,
        TERMINAL_DIRECTORY,
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


class V39StoreError(RuntimeError):
    """Fixed-code attempt-store rejection."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class V39StoreConflict(V39StoreError):
    """The attempt is already locked or an immutable destination exists."""


class V39StorePoisoned(V39StoreError):
    """The persisted attempt cannot be accepted or resumed."""


def _translate_journal_error(exc: V39JournalError) -> V39StorePoisoned:
    return V39StorePoisoned(exc.code)


def _validate_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(authority, Mapping):
        raise V39StoreError("authority_invalid")
    value = dict(authority)
    if frozenset(value) != _AUTHORITY_KEYS:
        raise V39StoreError("authority_invalid")
    attempt = value["attempt"]
    if not (
        attempt == ATTEMPT_ID
        or (
            type(attempt) is dict
            and attempt.get("attempt_id") == ATTEMPT_ID
        )
    ):
        raise V39StoreError("attempt_identity_invalid")
    try:
        # Round-trip once so tuples and mapping subclasses cannot create an
        # authority that writes canonically but compares differently on replay.
        normalized = json.loads(canonical_json_bytes(value))
    except (V39JournalError, json.JSONDecodeError):
        raise V39StoreError("authority_invalid") from None
    if type(normalized) is not dict:
        raise V39StoreError("authority_invalid")
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


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def _mkdir_new(path: Path) -> None:
    try:
        path.mkdir()
        secure_directory(path)
        fsync_directory(path.parent)
    except V39JournalError as exc:
        raise _translate_journal_error(exc) from None
    except OSError:
        raise V39StorePoisoned("attempt_directory_creation_failed") from None


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
    except V39JournalError as exc:
        raise V39StorePoisoned(exc.code) from None
    except OSError:
        raise V39StorePoisoned(code) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise V39StorePoisoned("attempt_root_invalid") from None
    return True


def _directory_details(path: Path, *, code: str) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError:
        raise V39StorePoisoned(code) from None
    if (
        not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
    ):
        raise V39StorePoisoned(code)
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
        raise V39StorePoisoned("attempt_initialization_failed")
    secure_directory(source.parent)
    secure_directory(destination.parent)
    source_details = _directory_details(
        source, code="attempt_initialization_failed"
    )
    if _entry_exists(destination):
        raise V39StoreConflict("attempt_already_exists")

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
    except V39StoreError:
        raise
    except (OSError, V39JournalError):
        if _entry_exists(destination):
            raise V39StoreConflict("attempt_already_exists") from None
        raise V39StorePoisoned("attempt_initialization_failed") from None
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
        raise V39StorePoisoned("attempt_initialization_failed")


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
            raise V39StoreConflict("attempt_already_exists")
        staging_parent = secure_directory(root.parent.parent)
        staging = Path(
            tempfile.mkdtemp(
                prefix=".v39-attempt-initialization-",
                dir=str(staging_parent),
            )
        ).resolve(strict=True)
        staging.relative_to(staging_parent)
        secure_directory(staging)
        _mkdir_new(staging / JOURNAL_DIRECTORY)
        _mkdir_new(staging / PAYLOAD_DIRECTORY)
        _mkdir_new(staging / CHECKPOINT_DIRECTORY)
        _mkdir_new(staging / TERMINAL_DIRECTORY)
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
            raise V39StorePoisoned("attempt_initialization_invalid")
        fsync_directory(staging)
        fsync_directory(staging_parent)
        if _entry_exists(root):
            raise V39StoreConflict("attempt_already_exists")
        _publish_attempt_root_noreplace(staging, root)
        fsync_directory(root.parent)
        fsync_directory(staging_parent)
    except (V39StoreError, V39JournalError):
        raise
    except (OSError, ValueError):
        if _entry_exists(root):
            raise V39StoreConflict("attempt_already_exists") from None
        raise V39StorePoisoned("attempt_initialization_failed") from None


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
                raise V39StoreConflict("attempt_locked")
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
                raise V39StorePoisoned("attempt_lock_identity_changed")
            handle.seek(0)
            if handle.read() != self.expected_bytes:
                raise V39StorePoisoned("attempt_lock_invalid")
            self._handle = handle
        except Exception as exc:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            with _ACTIVE_LOCKS_GUARD:
                _ACTIVE_LOCKS.discard(self.path)
            if isinstance(exc, V39StoreError):
                raise
            if isinstance(exc, V39JournalError):
                raise _translate_journal_error(exc) from None
            raise V39StoreConflict("attempt_locked") from None

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
class TerminalEvidenceReceipt:
    event_sha256: str
    evidence_sha256: str
    evidence_bytes: int
    journal_head_before_terminal_sha256: str
    status: str
    terminal_code: str
    evidence: Mapping[str, Any] = field(repr=False)


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
    )


def _exact_keys(payload: Mapping[str, Any], keys: set[str]) -> bool:
    return type(payload) is dict and set(payload) == keys


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
        raise V39StorePoisoned("checkpoint_subject_invalid")
    intent = machine.intents.get(
        response.payload["request_intent_event_sha256"]
    )
    if intent is None:
        raise V39StorePoisoned("checkpoint_subject_invalid")
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
        raise V39StorePoisoned("checkpoint_subject_invalid")
    segment = machine.segments[0]
    durations = tuple(segment.generation_durations_ns)
    if (
        segment.generation_count != PILOT_COUNT
        or len(durations) != PILOT_COUNT
        or any(type(item) is not int or item <= 0 for item in durations)
    ):
        raise V39StorePoisoned("pause_checkpoint_state_invalid")
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
        raise V39StorePoisoned("pause_checkpoint_state_invalid")
    return {
        "schema_version": PAUSE_CHECKPOINT_STATE_SCHEMA_VERSION,
        "phase": "paused_for_justification",
        "segment_id": segment.segment_id,
        "pilot_count": PILOT_COUNT,
        "pilot_durations_ns": list(durations),
        "projected_ns": projected,
        "threshold_ns": PAUSE_THRESHOLD_NS,
        "formula": "sum(pilot_duration_ns)+70*max(pilot_duration_ns)",
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
    raise V39StorePoisoned("checkpoint_subject_invalid")


def _validate_request_transition(machine: _Machine, intent: RequestIntent) -> None:
    if machine.terminal_event is not None:
        raise V39StorePoisoned("event_after_terminal")
    if machine.active_intent is not None or machine.pending_subject_event_sha256:
        raise V39StorePoisoned("request_boundary_incomplete")
    if machine.pause_committed and machine.continuation_event is None:
        raise V39StorePoisoned("request_while_paused")
    if intent.ordinal != len(machine.intents) + 1:
        raise V39StorePoisoned("request_order_invalid")
    if intent.request_id in machine.intent_ids:
        raise V39StorePoisoned("duplicate_request")

    if intent.effect_kind == "yahoo":
        if (
            intent.segment_id is not None
            or intent.model_phase != "none"
            or machine.open_segment_id is not None
            or machine.segments
            or machine.yahoo_responses >= YAHOO_REQUEST_COUNT
        ):
            raise V39StorePoisoned("yahoo_order_invalid")
        return

    if intent.effect_kind == "gemma":
        if (
            intent.model_phase != "generation"
            or intent.segment_id != machine.open_segment_id
            or machine.segment_phase != "generation"
            or machine.segment_pre_count != 2
        ):
            raise V39StorePoisoned("model_order_invalid")
        current = len(machine.segment_durations)
        expected = TOTAL_GENERATIONS if not machine.segments else REMAINING_COUNT
        if current >= expected:
            raise V39StorePoisoned("generation_count_exceeded")
        if not machine.segments and current == PILOT_COUNT:
            projection = _projected_ns(machine)
            if projection is None:
                raise V39StorePoisoned("pilot_timing_invalid")
            if projection > PAUSE_THRESHOLD_NS:
                raise V39StoreError("pilot_pause_required")
        return

    # Identity requests are the only effects that can open and close a model
    # segment.  Each probe is exactly version then show: two responses.
    if intent.effect_kind != "identity" or intent.segment_id is None:
        raise V39StorePoisoned("model_order_invalid")
    phase = intent.model_phase
    if phase == "pre_probe_start":
        if (
            machine.yahoo_responses != YAHOO_REQUEST_COUNT
            or machine.open_segment_id is not None
            or len(machine.segments) > 1
            or any(s.segment_id == intent.segment_id for s in machine.segments)
            or (machine.segments and machine.continuation_event is None)
            or (not machine.segments and machine.continuation_event is not None)
        ):
            raise V39StorePoisoned("model_segment_start_invalid")
        return
    if intent.segment_id != machine.open_segment_id:
        raise V39StorePoisoned("model_segment_mismatch")
    if phase == "pre_probe":
        if machine.segment_phase != "pre" or machine.segment_pre_count != 1:
            raise V39StorePoisoned("pre_probe_order_invalid")
        return
    if phase == "post_probe":
        if machine.segment_phase != "generation" or machine.segment_post_count:
            raise V39StorePoisoned("post_probe_order_invalid")
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
            raise V39StorePoisoned("post_probe_generation_count_invalid")
        return
    if phase == "post_probe_close":
        if machine.segment_phase != "post" or machine.segment_post_count != 1:
            raise V39StorePoisoned("post_probe_order_invalid")
        return
    raise V39StorePoisoned("model_phase_invalid")


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
        raise V39StorePoisoned("private_values_open_order_invalid")
    if event_type == "market_values_opened":
        if machine.market_values_opened or machine.model_responses_opened:
            raise V39StorePoisoned("private_values_open_order_invalid")
        return
    if event_type == "model_responses_opened":
        if not machine.market_values_opened or machine.model_responses_opened:
            raise V39StorePoisoned("private_values_open_order_invalid")
        return
    raise V39StorePoisoned("private_values_open_order_invalid")


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
        raise V39StorePoisoned("response_event_invalid")
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
        raise V39StorePoisoned("response_event_invalid")
    if active.effect_kind == "yahoo" and not (
        1 <= payload["body_bytes"] <= YAHOO_RESPONSE_CAP_BYTES
    ):
        raise V39StorePoisoned("yahoo_response_bytes_invalid")
    if active.effect_kind == "identity" and not (
        1 <= payload["body_bytes"] <= MODEL_RESPONSE_CAP_BYTES
    ):
        raise V39StorePoisoned("runtime_probe_response_bytes_invalid")
    if active.effect_kind == "gemma" and not (
        1 <= payload["body_bytes"] <= MODEL_RESPONSE_CAP_BYTES
    ):
        raise V39StorePoisoned("model_response_bytes_invalid")
    duration = payload["duration_ns"]
    if active.effect_kind == "gemma":
        if type(duration) is not int or duration <= 0:
            raise V39StorePoisoned("pilot_timing_invalid")
        machine.segment_durations.append(duration)
        machine.gemma_responses += 1
    elif duration is not None:
        raise V39StorePoisoned("response_duration_invalid")

    if active.effect_kind == "yahoo":
        machine.yahoo_responses += 1
        machine.yahoo_body_bytes += payload["body_bytes"]
        if machine.yahoo_body_bytes > YAHOO_BATCH_BODY_CAP_BYTES:
            raise V39StorePoisoned("yahoo_batch_body_cap_exceeded")
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
        raise V39StorePoisoned("attempt_intent_missing")
    for index, event in enumerate(replay.events):
        # A terminal marker closes the hash chain at the state-machine layer.
        # Checking this before event-type or payload dispatch makes the rule
        # universal: even an otherwise unknown or malformed event cannot be
        # interpreted after a terminal outcome.
        if machine.terminal_event is not None:
            raise V39StorePoisoned("event_after_terminal")
        if event.event_type not in _EVENT_TYPES:
            raise V39StorePoisoned("foreign_event")
        payload = event.payload
        if event.event_type == "attempt_intent":
            if (
                index != 0
                or not _exact_keys(payload, {"authority"})
                or payload["authority"] != dict(expected_authority)
            ):
                raise V39StorePoisoned("attempt_intent_invalid")
            continue
        if index == 0:
            raise V39StorePoisoned("attempt_intent_missing")

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
                raise V39StorePoisoned("request_intent_invalid")
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
                raise V39StorePoisoned("request_intent_invalid")
            _validate_request_transition(machine, intent)
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
                raise V39StorePoisoned("response_order_invalid")
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
                raise V39StorePoisoned("checkpoint_event_invalid")
            if (
                payload["subject_event_sha256"]
                != machine.pending_subject_event_sha256
                or payload["subject_type"] != machine.pending_subject_type
                or payload["subject_type"] not in {"response", "pause"}
                or not valid_sha256(payload["checkpoint_sha256"])
                or type(payload["checkpoint_bytes"]) is not int
                or payload["checkpoint_bytes"] <= 0
            ):
                raise V39StorePoisoned("checkpoint_event_invalid")
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
                },
            ):
                raise V39StorePoisoned("pause_event_invalid")
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
            ):
                raise V39StorePoisoned("pause_event_invalid")
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
                raise V39StorePoisoned("continuation_event_invalid")
            if (
                not machine.pause_committed
                or machine.pause_event is None
                or machine.continuation_event is not None
                or payload["pause_event_sha256"]
                != machine.pause_event.event_sha256
                or not valid_sha256(payload["continuation_sha256"])
                or not valid_sha256(payload["permission_sha256"])
                or not _valid_commit(payload["continuation_commit"])
            ):
                raise V39StorePoisoned("continuation_event_invalid")
            machine.continuation_event = event
            continue

        if event.event_type in {
            "market_values_opened",
            "model_responses_opened",
        }:
            if not _exact_keys(payload, set()):
                raise V39StorePoisoned("private_values_open_event_invalid")
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
                    "evidence_sha256",
                    "evidence_bytes",
                    "journal_head_before_terminal_sha256",
                },
            ):
                raise V39StorePoisoned("terminal_event_invalid")
            if (
                machine.terminal_event is not None
                or payload["status"]
                not in {"completed", "rejected", "indeterminate"}
                or not _safe_code(payload["terminal_code"])
                or not valid_sha256(payload["evidence_sha256"])
                or type(payload["evidence_bytes"]) is not int
                or payload["evidence_bytes"] <= 0
                or payload["journal_head_before_terminal_sha256"]
                != event.previous_event_sha256
                or machine.pause_committed
                and machine.continuation_event is None
                or machine.active_intent is not None
                and payload["status"] != "indeterminate"
            ):
                raise V39StorePoisoned("terminal_event_invalid")
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
                    raise V39StorePoisoned("completed_effect_budget_invalid")
            machine.terminal_event = event
            machine.terminal_code = payload["terminal_code"]
            continue

    return machine


def _decode_canonical_file(path: Path, *, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        details = regular_file(path, code=code)
        if details.st_size <= 0 or details.st_size > _MAX_CONTENT_BYTES:
            raise V39StorePoisoned(code)
        encoded = read_regular_bytes(path, code=code)
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except V39StoreError:
        raise
    except (V39JournalError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise V39StorePoisoned(code) from None
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise V39StorePoisoned(code)
    return value, encoded


def _scan_content(directory: Path, *, code: str) -> dict[str, tuple[dict[str, Any], bytes]]:
    try:
        secure_directory(directory)
        entries = tuple(directory.iterdir())
    except V39JournalError as exc:
        raise _translate_journal_error(exc) from None
    except OSError:
        raise V39StorePoisoned(code) from None
    result: dict[str, tuple[dict[str, Any], bytes]] = {}
    for path in entries:
        match = _CONTENT_NAME_RE.fullmatch(path.name)
        if match is None:
            raise V39StorePoisoned(code)
        value, encoded = _decode_canonical_file(path, code=code)
        digest = sha256_bytes(encoded)
        if digest != match.group(1) or digest in result:
            raise V39StorePoisoned(code)
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
        raise V39StorePoisoned("response_payload_missing")
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
        raise V39StorePoisoned("response_payload_invalid")
    try:
        body = base64.b64decode(value["body_base64"], validate=True)
    except (ValueError, TypeError):
        raise V39StorePoisoned("response_payload_invalid") from None
    if len(body) != value["body_bytes"] or sha256_bytes(body) != value["body_sha256"]:
        raise V39StorePoisoned("response_payload_invalid")
    intent = machine.intents.get(event_payload["request_intent_event_sha256"])
    if intent is None:
        raise V39StorePoisoned("response_payload_invalid")
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
        raise V39StorePoisoned("orphan_response_payload")

    referenced_checkpoints: set[str] = set()
    for subject, checkpoint_event in machine.checkpoints.items():
        digest = checkpoint_event.payload["checkpoint_sha256"]
        stored = checkpoints.get(digest)
        if stored is None:
            raise V39StorePoisoned("checkpoint_missing")
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
            raise V39StorePoisoned("checkpoint_invalid")
        referenced_checkpoints.add(digest)

    candidates = tuple(sorted(set(checkpoints) - referenced_checkpoints))
    if candidates:
        # Exactly one marker-last candidate is permitted only while the journal
        # has one authoritative response/pause awaiting its checkpoint marker.
        if machine.pending_subject_event_sha256 is None or len(candidates) != 1:
            raise V39StorePoisoned("orphan_checkpoint")
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
            raise V39StorePoisoned("orphan_checkpoint")
    return loaded, checkpoints, candidates


def _audit_terminal_evidence(
    machine: _Machine, terminal_dir: Path
) -> TerminalEvidenceReceipt | None:
    stored = _scan_content(
        terminal_dir, code="terminal_evidence_directory_poisoned"
    )
    event = machine.terminal_event
    if event is None:
        if stored:
            raise V39StorePoisoned("orphan_terminal_evidence")
        return None
    digest = event.payload["evidence_sha256"]
    if set(stored) != {digest}:
        if digest not in stored:
            raise V39StorePoisoned("terminal_evidence_missing")
        raise V39StorePoisoned("orphan_terminal_evidence")
    value, encoded = stored[digest]
    if (
        len(encoded) != event.payload["evidence_bytes"]
        or set(value)
        != {
            "schema_version",
            "authority_sha256",
            "journal_head_before_terminal_sha256",
            "status",
            "terminal_code",
            "evidence",
        }
        or value["schema_version"] != TERMINAL_EVIDENCE_SCHEMA_VERSION
        or value["authority_sha256"] != machine.authority_sha256
        or value["journal_head_before_terminal_sha256"]
        != event.payload["journal_head_before_terminal_sha256"]
        or value["status"] != event.payload["status"]
        or value["terminal_code"] != event.payload["terminal_code"]
        or type(value["evidence"]) is not dict
    ):
        raise V39StorePoisoned("terminal_evidence_invalid")
    return TerminalEvidenceReceipt(
        event.event_sha256,
        digest,
        len(encoded),
        value["journal_head_before_terminal_sha256"],
        value["status"],
        value["terminal_code"],
        _freeze_json(value["evidence"]),
    )


def _root_layout(root: Path) -> None:
    try:
        secure_directory(root)
        names = frozenset(path.name for path in root.iterdir())
    except V39JournalError as exc:
        raise _translate_journal_error(exc) from None
    except OSError:
        raise V39StorePoisoned("attempt_root_invalid") from None
    if names != _ROOT_ENTRIES:
        raise V39StorePoisoned("attempt_root_poisoned")


def _content_bytes(subject_event_sha256: str, state: Mapping[str, Any]) -> bytes:
    if not isinstance(state, Mapping):
        raise V39StoreError("checkpoint_state_invalid")
    value = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "subject_event_sha256": subject_event_sha256,
        "state": dict(state),
    }
    return canonical_json_bytes(value)


class AttemptStore:
    """Exclusively own and advance one immutable v3.9 attempt namespace."""

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
        self._machine: _Machine | None = None
        self._responses: dict[str, CommittedResponse] = {}
        self._checkpoints: dict[str, tuple[dict[str, Any], bytes]] = {}
        self._checkpoint_candidates: tuple[str, ...] = ()
        self._terminal_evidence: TerminalEvidenceReceipt | None = None
        self._live_intent_hashes: set[str] = set()
        self._live_model_segments: set[str] = set()
        self._live_pending_subjects: set[str] = set()
        self._recovering_subject: str | None = None

        if not isinstance(root, Path) or not root.is_absolute():
            raise V39StoreError("attempt_root_invalid")
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
            raise V39StoreError("store_closed")
        if self._poisoned:
            raise V39StorePoisoned("store_poisoned")

    def privacy_scan_locked_payloads(self) -> Mapping[str, bytes]:
        """Return exact bytes for files that this process locks against reread."""

        self._require_open()
        if self._lock is None:
            raise V39StoreError("attempt_lock_invalid")
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
            terminal_evidence = _audit_terminal_evidence(
                machine, self.root / TERMINAL_DIRECTORY
            )
            self._machine = machine
            self._responses = responses
            self._checkpoints = checkpoints
            self._checkpoint_candidates = candidates
            self._terminal_evidence = terminal_evidence
        except V39JournalError as exc:
            self._poisoned = True
            raise _translate_journal_error(exc) from None
        except V39StorePoisoned:
            self._poisoned = True
            raise

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> JournalEvent:
        self._require_open()
        if self._journal is None:
            raise V39StoreError("store_not_initialized")
        try:
            event = self._journal.append(event_type, payload)
            self._refresh()
            return event
        except V39JournalError as exc:
            self._poisoned = True
            raise _translate_journal_error(exc) from None

    def _machine_or_raise(self) -> _Machine:
        self._require_open()
        if self._machine is None:
            raise V39StoreError("store_not_initialized")
        return self._machine

    @property
    def snapshot(self) -> AttemptSnapshot:
        machine = self._machine_or_raise()
        status = "active"
        terminal_code = machine.terminal_code
        recoverable = False
        if machine.terminal_event is not None:
            status = str(machine.terminal_event.payload["status"])
        elif machine.active_intent is not None:
            live = machine.active_intent.event_sha256 in self._live_intent_hashes
            status = "active" if live else "indeterminate"
            terminal_code = None if live else "open_request_intent"
        elif machine.open_segment_id is not None:
            live = machine.open_segment_id in self._live_model_segments
            status = "active" if live else "indeterminate"
            terminal_code = None if live else "model_segment_interrupted"
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
                terminal_code = "model_segment_interrupted"
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
            raise V39StoreError("request_not_authorized")
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
            raise V39StoreError("request_intent_invalid")
        _validate_request_transition(machine, candidate)
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
            raise V39StoreError("response_commit_invalid")
        if active.effect_kind == "yahoo" and not (
            1 <= len(body) <= YAHOO_RESPONSE_CAP_BYTES
        ):
            raise V39StoreError("yahoo_response_bytes_invalid")
        if active.effect_kind == "identity" and not (
            1 <= len(body) <= MODEL_RESPONSE_CAP_BYTES
        ):
            raise V39StoreError("runtime_probe_response_bytes_invalid")
        if active.effect_kind == "gemma" and not (
            1 <= len(body) <= MODEL_RESPONSE_CAP_BYTES
        ):
            raise V39StoreError("model_response_bytes_invalid")
        payload_value = {
            "schema_version": RESPONSE_PAYLOAD_SCHEMA_VERSION,
            "request_intent_event_sha256": active.event_sha256,
            "request_id": active.request_id,
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_bytes": len(body),
            "body_sha256": sha256_bytes(body),
            "metadata": dict(metadata),
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
            raise V39StoreError("content_size_invalid")
        if sha256_bytes(encoded) != digest:
            raise V39StoreError("content_hash_invalid")
        final = directory / f"{digest}.json"
        pending = directory / f".pending-{digest}.json"
        try:
            final_exists = entry_exists_nofollow(
                final, code="content_write_failed"
            )
            pending_exists = entry_exists_nofollow(
                pending, code="content_write_failed"
            )
        except V39JournalError as exc:
            self._poisoned = True
            raise V39StorePoisoned(exc.code) from None
        if final_exists:
            try:
                regular_file(final, code="content_write_failed")
            except V39JournalError as exc:
                self._poisoned = True
                raise V39StorePoisoned(exc.code) from None
            if allow_existing_exact:
                try:
                    if (
                        read_regular_bytes(
                            final, code="content_write_failed"
                        )
                        == encoded
                    ):
                        return final
                except V39JournalError as exc:
                    self._poisoned = True
                    raise V39StorePoisoned(exc.code) from None
            raise V39StoreConflict(conflict_code)
        if pending_exists:
            raise V39StorePoisoned("content_write_incomplete")
        try:
            _write_exclusive(pending, encoded, code="content_write_failed")
            try:
                publish_noreplace(
                    pending, final, code="content_destination_exists"
                )
            except V39JournalError as exc:
                try:
                    destination_exists = entry_exists_nofollow(
                        final, code="content_write_failed"
                    )
                except V39JournalError as presence_exc:
                    raise V39StorePoisoned(presence_exc.code) from None
                if destination_exists:
                    raise V39StoreConflict(conflict_code) from None
                raise V39StorePoisoned(exc.code) from None
            fsync_directory(directory)
            _value, reread = _decode_canonical_file(
                final, code="content_write_failed"
            )
            if reread != encoded:
                raise V39StorePoisoned("content_write_failed")
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
            raise V39StoreError("checkpoint_subject_invalid")
        expected_state = _derive_checkpoint_state(
            machine,
            subject_event_sha256,
            machine.pending_subject_type,
        )
        if not isinstance(state, Mapping) or dict(state) != expected_state:
            raise V39StoreError("checkpoint_state_invalid")
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
            raise V39StoreError("checkpoint_not_pending")
        return _derive_checkpoint_state(machine, subject, subject_type)

    def pending_checkpoint_context(self) -> PendingCheckpointContext:
        machine = self._machine_or_raise()
        subject = machine.pending_subject_event_sha256
        if subject is None:
            raise V39StoreError("checkpoint_not_pending")
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
            raise V39StoreError("response_filter_invalid")
        if segment_id is not None and not _safe_id(segment_id):
            raise V39StoreError("response_filter_invalid")
        if effect_kind == "yahoo" and segment_id is not None:
            raise V39StoreError("response_filter_invalid")

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
            raise V39StoreError("response_batch_unsealed")

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
                raise V39StoreError("response_batch_unsealed")
            if not (
                machine.market_values_opened
                and machine.model_responses_opened
            ):
                raise V39StoreError("response_values_not_opened")
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
                raise V39StoreError("response_batch_unsealed")
            if not machine.market_values_opened:
                raise V39StoreError("response_values_not_opened")
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
                raise V39StoreError("response_batch_unsealed")
            segment_ids = {item.segment_id for item in machine.segments}
            if segment_id is not None and segment_id not in segment_ids:
                raise V39StoreError("response_filter_invalid")
            for summary in machine.segments:
                if summary.close_response_event_sha256 not in machine.checkpoints:
                    raise V39StoreError("response_batch_unsealed")
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
                raise V39StoreError("response_batch_unsealed")
            if not machine.model_responses_opened:
                raise V39StoreError("response_values_not_opened")
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
                raise V39StorePoisoned("identity_segment_missing")
            groups.setdefault(identity_segment, []).append(response)
        if segment_id is not None and segment_id not in groups:
            raise V39StoreError("response_filter_invalid")
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
                raise V39StoreError("response_batch_unsealed")
            if len(responses) != expected or any(
                response.event_sha256 not in machine.checkpoints
                for response in responses
            ):
                raise V39StoreError("response_batch_unsealed")
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
            raise V39StoreError("response_filter_invalid")
        if segment_id is not None and not _safe_id(segment_id):
            raise V39StoreError("response_filter_invalid")
        if effect_kind == "yahoo" and segment_id is not None:
            raise V39StoreError("response_filter_invalid")
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
            raise V39StoreError("response_batch_unsealed")

        if effect_kind == "yahoo":
            if machine.yahoo_responses != YAHOO_REQUEST_COUNT or len(selected) != YAHOO_REQUEST_COUNT:
                raise V39StoreError("response_batch_unsealed")
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
                raise V39StoreError("response_filter_invalid")
            for target_id in target_ids:
                summary = summaries.get(target_id)
                if (
                    summary is None
                    or summary.close_response_event_sha256
                    not in machine.checkpoints
                ):
                    raise V39StoreError("response_batch_unsealed")
                count = sum(
                    response.intent.segment_id == target_id
                    for response in selected
                )
                if count != summary.generation_count:
                    raise V39StoreError("response_batch_unsealed")

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
            raise V39StoreError("checkpoint_builder_invalid")
        snapshot = self.snapshot
        if snapshot.status != "checkpoint_recovery":
            raise V39StoreError("checkpoint_recovery_forbidden")
        context = self.pending_checkpoint_context()
        try:
            rebuilt = builder(context)
        except V39StoreError:
            raise
        except Exception:
            raise V39StoreError("checkpoint_rebuild_failed") from None
        if not isinstance(rebuilt, Mapping):
            raise V39StoreError("checkpoint_rebuild_failed")
        expected = dict(self.derive_pending_checkpoint_state())
        if dict(rebuilt) != expected:
            raise V39StoreError("checkpoint_rebuild_failed")
        encoded = _content_bytes(context.subject_event_sha256, expected)
        digest = sha256_bytes(encoded)
        if self._checkpoint_candidates:
            if self._checkpoint_candidates != (digest,):
                self._poisoned = True
                raise V39StorePoisoned("checkpoint_mismatch")
            stored = self._checkpoints[digest][1]
            if stored != encoded:
                self._poisoned = True
                raise V39StorePoisoned("checkpoint_mismatch")
        self._recovering_subject = context.subject_event_sha256
        try:
            return self.commit_checkpoint(
                subject_event_sha256=context.subject_event_sha256,
                state=expected,
            )
        finally:
            self._recovering_subject = None

    def commit_planned_pause(
        self, *, checkpoint_state: Mapping[str, Any] | None = None
    ) -> tuple[JournalEvent, JournalEvent]:
        machine = self._machine_or_raise()
        if (
            machine.active_intent is not None
            or machine.pending_subject_event_sha256 is not None
            or machine.open_segment_id is not None
            or len(machine.segments) != 1
            or machine.segments[0].generation_count != PILOT_COUNT
            or machine.pause_event is not None
        ):
            raise V39StoreError("pause_not_authorized")
        durations = machine.segments[0].generation_durations_ns
        projected = sum(durations) + REMAINING_COUNT * max(durations)
        if projected <= PAUSE_THRESHOLD_NS:
            raise V39StoreError("pause_threshold_not_crossed")

        pause_payload_base = {
            "pilot_count": PILOT_COUNT,
            "pilot_durations_ns": list(durations),
            "projected_ns": projected,
            "threshold_ns": PAUSE_THRESHOLD_NS,
            "formula": "sum(pilot_duration_ns)+70*max(pilot_duration_ns)",
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
            raise V39StoreError("checkpoint_state_invalid")
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
        if (
            not machine.pause_committed
            or machine.pause_event is None
            or machine.continuation_event is not None
            or not valid_sha256(continuation_sha256)
            or not valid_sha256(permission_sha256)
            or not _valid_commit(continuation_commit)
        ):
            raise V39StoreError("continuation_not_authorized")
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
        except V39StorePoisoned:
            raise V39StoreError("private_values_open_not_authorized") from None
        return self._append("market_values_opened", {})

    def mark_model_responses_opened(self) -> JournalEvent:
        """Durably record the conservative boundary before model parsing."""

        machine = self._machine_or_raise()
        try:
            _validate_private_open_transition(machine, "model_responses_opened")
        except V39StorePoisoned:
            raise V39StoreError("private_values_open_not_authorized") from None
        return self._append("model_responses_opened", {})

    def seal_terminal(
        self,
        *,
        status: str,
        terminal_code: str,
        evidence: Mapping[str, Any],
    ) -> TerminalEvidenceReceipt:
        """Persist private evidence first, then bind it marker-last to terminal."""

        machine = self._machine_or_raise()
        if (
            status not in {"completed", "rejected", "indeterminate"}
            or not _safe_code(terminal_code)
            or not isinstance(evidence, Mapping)
            or machine.terminal_event is not None
            or machine.pause_committed
            and machine.continuation_event is None
            or machine.active_intent is not None
            and status != "indeterminate"
        ):
            raise V39StoreError("terminal_invalid")
        if status == "completed":
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
                raise V39StoreError("completed_effect_budget_invalid")
        prefix = machine.events[-1].event_sha256
        envelope = {
            "schema_version": TERMINAL_EVIDENCE_SCHEMA_VERSION,
            "authority_sha256": self.authority_sha256,
            "journal_head_before_terminal_sha256": prefix,
            "status": status,
            "terminal_code": terminal_code,
            "evidence": dict(evidence),
        }
        encoded = canonical_json_bytes(envelope)
        digest = sha256_bytes(encoded)
        self._write_content(
            self.root / TERMINAL_DIRECTORY,
            digest,
            encoded,
            allow_existing_exact=False,
            conflict_code="terminal_evidence_exists",
        )
        try:
            self._append(
                "attempt_terminal",
                {
                    "status": status,
                    "terminal_code": terminal_code,
                    "evidence_sha256": digest,
                    "evidence_bytes": len(encoded),
                    "journal_head_before_terminal_sha256": prefix,
                },
            )
        except Exception:
            self._poisoned = True
            raise
        if self._terminal_evidence is None:
            self._poisoned = True
            raise V39StorePoisoned("terminal_evidence_missing")
        return self._terminal_evidence

    def committed_terminal_evidence(self) -> TerminalEvidenceReceipt:
        """Return the replay-audited private terminal evidence, if committed."""

        self._require_open()
        if self._terminal_evidence is None:
            raise V39StoreError("terminal_evidence_not_committed")
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
    "PAUSE_CHECKPOINT_STATE_SCHEMA_VERSION",
    "PendingResponseReceipt",
    "PILOT_COUNT",
    "PendingCheckpointContext",
    "REMAINING_COUNT",
    "RESPONSE_PAYLOAD_SCHEMA_VERSION",
    "RESPONSE_CHECKPOINT_STATE_SCHEMA_VERSION",
    "RequestIntent",
    "SegmentSummary",
    "TERMINAL_EVIDENCE_SCHEMA_VERSION",
    "TerminalEvidenceReceipt",
    "TOTAL_GENERATIONS",
    "V39StoreConflict",
    "V39StoreError",
    "V39StorePoisoned",
    "YAHOO_REQUEST_COUNT",
    "YAHOO_BATCH_BODY_CAP_BYTES",
    "audit_attempt",
]
