"""Canonical append-only journal used by the v3.13 science attempt.

This module is deliberately boring: it knows how to durably append and replay
one hash chain, but it knows nothing about SEC, Yahoo, Ollama, or trading.  The
state-machine rules live in :mod:`sec_gemma_lean_science_v313_store`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Final


JOURNAL_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-13-journal-v1"
)
ZERO_SHA256: Final[str] = "0" * 64
MAX_EVENT_BYTES: Final[int] = 4 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EVENT_TYPE_RE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_EVENT_NAME_RE = re.compile(r"([0-9]{8})-([0-9a-f]{64})\.json\Z")
_PENDING_NAME_RE = re.compile(
    r"\.pending-([0-9]{8})-([0-9a-f]{64})\.json\Z"
)
_FAILED_NAME_RE = re.compile(
    r"\.append-failed-([0-9]{8})-([0-9a-f]{64})\.json\Z"
)
_REPARSE_ATTRIBUTE: Final[int] = 0x400


class V313JournalError(RuntimeError):
    """A fixed-code journal error that contains no persisted third-party text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _validate_json_value(value: Any) -> None:
    """Reject values whose JSON spelling is ambiguous or platform-dependent."""

    if value is None or type(value) in {bool, int, str}:
        if type(value) is str:
            try:
                value.encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                raise V313JournalError("canonical_json_invalid") from None
        return
    # Durable binary64 values in this experiment are hexadecimal strings.  A
    # native float in state is therefore always a caller error, even if finite.
    if type(value) is float:
        raise V313JournalError("canonical_json_float_forbidden")
    if type(value) in {list, tuple}:
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise V313JournalError("canonical_json_invalid")
            _validate_json_value(key)
            _validate_json_value(item)
        return
    raise V313JournalError("canonical_json_invalid")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the preregistered UTF-8 canonical JSON representation.

    V3.13 removed the historical trailing newline from every canonical JSON
    preimage. Journal filenames and hash links bind these exact bytes.
    """

    _validate_json_value(value)
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError):
        raise V313JournalError("canonical_json_invalid") from None


def sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise V313JournalError("bytes_invalid")
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def valid_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def secure_directory(path: Path) -> Path:
    """Require an absolute, existing, non-link directory path component-wise."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise V313JournalError("journal_directory_invalid")
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor /= component
        try:
            details = cursor.lstat()
        except OSError:
            raise V313JournalError("journal_directory_invalid") from None
        if (
            not stat.S_ISDIR(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or _is_reparse(details)
        ):
            raise V313JournalError("journal_directory_invalid")
    return path


def regular_file(path: Path, *, code: str = "journal_invalid") -> os.stat_result:
    try:
        details = path.lstat()
    except OSError:
        raise V313JournalError(code) from None
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
    ):
        raise V313JournalError(code)
    return details


def entry_exists_nofollow(path: Path, *, code: str = "journal_invalid") -> bool:
    """Test directory-entry presence without ever resolving the last component."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise V313JournalError(code) from None
    return True


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare identities conservatively across CPython's POSIX/Windows stats."""

    left_ino = getattr(left, "st_ino", 0)
    right_ino = getattr(right, "st_ino", 0)
    return (
        left.st_dev == right.st_dev
        and left_ino != 0
        and right_ino != 0
        and left_ino == right_ino
    )


def _regular_descriptor(
    descriptor: int, *, code: str
) -> os.stat_result:
    try:
        details = os.fstat(descriptor)
    except OSError:
        raise V313JournalError(code) from None
    if (
        not stat.S_ISREG(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
    ):
        raise V313JournalError(code)
    return details


def _open_exclusive_regular(path: Path, *, code: str) -> int:
    """Create one new regular file without following a final-component link."""

    secure_directory(path.parent)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError:
        raise V313JournalError(code) from None
    try:
        _regular_descriptor(descriptor, code=code)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def read_regular_bytes(path: Path, *, code: str = "journal_invalid") -> bytes:
    """Read one immutable regular file through a no-follow descriptor."""

    secure_directory(path.parent)
    before = regular_file(path, code=code)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = _regular_descriptor(descriptor, code=code)
        if not _same_file_identity(before, opened):
            raise V313JournalError(code)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = _regular_descriptor(descriptor, code=code)
        visible = regular_file(path, code=code)
        if (
            not _same_file_identity(opened, after)
            or not _same_file_identity(opened, visible)
            or after.st_size != sum(len(chunk) for chunk in chunks)
        ):
            raise V313JournalError(code)
        return b"".join(chunks)
    except V313JournalError:
        raise
    except OSError:
        raise V313JournalError(code) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def publish_noreplace(source: Path, destination: Path, *, code: str) -> None:
    """Publish a regular staging file atomically without replacing any entry.

    Windows ``rename`` already refuses an existing destination.  On POSIX a
    hard-link publication provides the same no-replace property; removing the
    staging name afterwards leaves the destination with exactly one link.
    Neither branch follows a source or destination symlink.
    """

    if source.parent != destination.parent:
        raise V313JournalError(code)
    secure_directory(source.parent)
    source_details = regular_file(source, code=code)
    if entry_exists_nofollow(destination, code=code):
        raise V313JournalError(code)
    try:
        if os.name == "nt":
            os.rename(source, destination)
        else:
            os.link(source, destination, follow_symlinks=False)
            linked = destination.lstat()
            source_linked = source.lstat()
            if (
                not stat.S_ISREG(linked.st_mode)
                or stat.S_ISLNK(linked.st_mode)
                or _is_reparse(linked)
                or not _same_file_identity(source_linked, linked)
                or source_linked.st_nlink != 2
                or linked.st_nlink != 2
            ):
                raise OSError("publication_identity")
            os.unlink(source)
        published = regular_file(destination, code=code)
        if not _same_file_identity(source_details, published):
            raise V313JournalError(code)
    except V313JournalError:
        raise
    except OSError:
        raise V313JournalError(code) from None


def fsync_directory(directory: Path) -> None:
    """Durably flush a directory on POSIX and Windows where supported."""

    secure_directory(directory)
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create = kernel32.CreateFileW
            create.argtypes = (
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            )
            create.restype = wintypes.HANDLE
            flush = kernel32.FlushFileBuffers
            flush.argtypes = (wintypes.HANDLE,)
            flush.restype = wintypes.BOOL
            close = kernel32.CloseHandle
            close.argtypes = (wintypes.HANDLE,)
            close.restype = wintypes.BOOL
            handle = create(
                str(directory),
                0x40000000,  # GENERIC_WRITE
                7,  # FILE_SHARE_READ | WRITE | DELETE
                None,
                3,  # OPEN_EXISTING
                0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
                None,
            )
            if handle == ctypes.c_void_p(-1).value:
                raise OSError("open")
            try:
                if not flush(handle):
                    raise OSError("flush")
            finally:
                close(handle)
            return
        except (AttributeError, OSError):
            raise V313JournalError("journal_directory_fsync_failed") from None

    descriptor: int | None = None
    try:
        descriptor = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        os.fsync(descriptor)
    except OSError:
        raise V313JournalError("journal_directory_fsync_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_exclusive_fsynced(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = _open_exclusive_regular(
            path, code="journal_append_durability_failed"
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short_write")
            offset += written
        os.fsync(descriptor)
        details = _regular_descriptor(
            descriptor, code="journal_append_durability_failed"
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        reread = bytearray()
        while len(reread) < len(payload):
            chunk = os.read(descriptor, len(payload) - len(reread))
            if not chunk:
                break
            reread.extend(chunk)
        visible = regular_file(
            path, code="journal_append_durability_failed"
        )
        if (
            details.st_size != len(payload)
            or bytes(reread) != payload
            or not _same_file_identity(details, visible)
        ):
            raise OSError("changed")
    except V313JournalError:
        raise
    except OSError:
        raise V313JournalError("journal_append_durability_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _leave_failure_marker(root: Path, sequence: int, event_sha256: str) -> None:
    marker = root / f".append-failed-{sequence:08d}-{event_sha256}.json"
    try:
        if not entry_exists_nofollow(marker):
            _write_exclusive_fsynced(
                marker,
                canonical_json_bytes(
                    {
                        "schema_version": JOURNAL_SCHEMA_VERSION,
                        "sequence": sequence,
                        "event_sha256": event_sha256,
                        "error_code": "journal_append_durability_failed",
                    }
                ),
            )
        fsync_directory(root)
    except Exception:
        # A visible marker already forces replay to fail closed.  It must never
        # be removed merely because the second durability operation failed.
        return


@dataclass(frozen=True)
class JournalEvent:
    sequence: int
    event_type: str
    authority_sha256: str
    previous_event_sha256: str
    payload: Mapping[str, Any] = field(repr=False)
    event_sha256: str
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "authority_sha256": self.authority_sha256,
            "previous_event_sha256": self.previous_event_sha256,
            "payload": dict(self.payload),
            "event_sha256": self.event_sha256,
        }


@dataclass(frozen=True)
class JournalReplay:
    authority_sha256: str
    events: tuple[JournalEvent, ...]

    @property
    def head_event_sha256(self) -> str:
        return self.events[-1].event_sha256 if self.events else ZERO_SHA256

    @property
    def event_count(self) -> int:
        return len(self.events)


_EVENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "sequence",
        "event_type",
        "authority_sha256",
        "previous_event_sha256",
        "payload",
        "event_sha256",
    }
)


def _decode_event(path: Path, *, expected_authority_sha256: str) -> JournalEvent:
    details = regular_file(path)
    if details.st_size <= 0 or details.st_size > MAX_EVENT_BYTES:
        raise V313JournalError("journal_invalid")
    try:
        encoded = read_regular_bytes(path)
        # Recheck the bytes actually read.  The directory entry can be replaced
        # between the preliminary size check above and the descriptor-safe
        # read, so the first stat alone is not a trustworthy size bound.
        if not encoded or len(encoded) > MAX_EVENT_BYTES:
            raise V313JournalError("journal_invalid")
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except V313JournalError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise V313JournalError("journal_invalid") from None
    if type(value) is not dict or frozenset(value) != _EVENT_KEYS:
        raise V313JournalError("journal_invalid")
    if canonical_json_bytes(value) != encoded:
        raise V313JournalError("journal_noncanonical")
    if (
        value["schema_version"] != JOURNAL_SCHEMA_VERSION
        or type(value["sequence"]) is not int
        or value["sequence"] <= 0
        or type(value["event_type"]) is not str
        or _EVENT_TYPE_RE.fullmatch(value["event_type"]) is None
        or value["authority_sha256"] != expected_authority_sha256
        or not valid_sha256(value["previous_event_sha256"])
        or type(value["payload"]) is not dict
        or not valid_sha256(value["event_sha256"])
    ):
        raise V313JournalError("journal_invalid")
    unsigned = {key: item for key, item in value.items() if key != "event_sha256"}
    calculated = canonical_sha256(unsigned)
    match = _EVENT_NAME_RE.fullmatch(path.name)
    if (
        calculated != value["event_sha256"]
        or match is None
        or int(match.group(1)) != value["sequence"]
        or match.group(2) != value["event_sha256"]
    ):
        raise V313JournalError("journal_hash_mismatch")
    return JournalEvent(
        value["sequence"],
        value["event_type"],
        value["authority_sha256"],
        value["previous_event_sha256"],
        value["payload"],
        value["event_sha256"],
        path,
    )


def replay_journal(root: Path, *, authority_sha256: str) -> JournalReplay:
    """Strictly replay an exact, gap-free chain; any extra entry poisons it."""

    secure_directory(root)
    if not valid_sha256(authority_sha256):
        raise V313JournalError("authority_sha256_invalid")
    try:
        entries = tuple(root.iterdir())
    except OSError:
        raise V313JournalError("journal_invalid") from None
    parsed_names: list[tuple[int, Path]] = []
    for entry in entries:
        if _PENDING_NAME_RE.fullmatch(entry.name) or _FAILED_NAME_RE.fullmatch(
            entry.name
        ):
            raise V313JournalError("journal_append_incomplete")
        match = _EVENT_NAME_RE.fullmatch(entry.name)
        if match is None:
            raise V313JournalError("journal_extra_entry")
        parsed_names.append((int(match.group(1)), entry))
    parsed_names.sort(key=lambda item: item[0])
    if [sequence for sequence, _path in parsed_names] != list(
        range(1, len(parsed_names) + 1)
    ):
        raise V313JournalError("journal_sequence_invalid")

    events: list[JournalEvent] = []
    previous = ZERO_SHA256
    for sequence, path in parsed_names:
        event = _decode_event(
            path, expected_authority_sha256=authority_sha256
        )
        if event.sequence != sequence or event.previous_event_sha256 != previous:
            raise V313JournalError("journal_chain_invalid")
        events.append(event)
        previous = event.event_sha256
    return JournalReplay(authority_sha256, tuple(events))


class HashChainJournal:
    """Append events to one already-created journal directory."""

    def __init__(self, root: Path, *, authority_sha256: str) -> None:
        self.root = secure_directory(root)
        if not valid_sha256(authority_sha256):
            raise V313JournalError("authority_sha256_invalid")
        self.authority_sha256 = authority_sha256
        replay_journal(self.root, authority_sha256=self.authority_sha256)

    def replay(self) -> JournalReplay:
        return replay_journal(
            self.root, authority_sha256=self.authority_sha256
        )

    def append(self, event_type: str, payload: Mapping[str, Any]) -> JournalEvent:
        if (
            type(event_type) is not str
            or _EVENT_TYPE_RE.fullmatch(event_type) is None
            or not isinstance(payload, Mapping)
        ):
            raise V313JournalError("journal_event_invalid")
        # Normalize mapping subclasses now so the bytes replay as a built-in dict.
        normalized_payload = dict(payload)
        canonical_json_bytes(normalized_payload)
        replay = self.replay()
        sequence = replay.event_count + 1
        unsigned = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "sequence": sequence,
            "event_type": event_type,
            "authority_sha256": self.authority_sha256,
            "previous_event_sha256": replay.head_event_sha256,
            "payload": normalized_payload,
        }
        event_sha256 = canonical_sha256(unsigned)
        body = {**unsigned, "event_sha256": event_sha256}
        encoded = canonical_json_bytes(body)
        if len(encoded) > MAX_EVENT_BYTES:
            raise V313JournalError("journal_event_too_large")
        final = self.root / f"{sequence:08d}-{event_sha256}.json"
        pending = self.root / f".pending-{sequence:08d}-{event_sha256}.json"
        if entry_exists_nofollow(final) or entry_exists_nofollow(pending):
            raise V313JournalError("journal_destination_exists")

        try:
            _write_exclusive_fsynced(pending, encoded)
            publish_noreplace(
                pending, final, code="journal_destination_exists"
            )
            fsync_directory(self.root)
        except Exception as exc:
            _leave_failure_marker(self.root, sequence, event_sha256)
            if isinstance(exc, V313JournalError):
                raise
            raise V313JournalError("journal_append_durability_failed") from None

        committed = _decode_event(
            final, expected_authority_sha256=self.authority_sha256
        )
        if committed.previous_event_sha256 != replay.head_event_sha256:
            raise V313JournalError("journal_chain_invalid")
        return committed


__all__ = [
    "HashChainJournal",
    "JOURNAL_SCHEMA_VERSION",
    "JournalEvent",
    "JournalReplay",
    "V313JournalError",
    "ZERO_SHA256",
    "canonical_json_bytes",
    "canonical_sha256",
    "entry_exists_nofollow",
    "fsync_directory",
    "publish_noreplace",
    "read_regular_bytes",
    "regular_file",
    "replay_journal",
    "secure_directory",
    "sha256_bytes",
    "valid_sha256",
]
