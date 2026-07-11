"""Deterministic, filesystem-only sealing for flat SEC audit artifacts.

The caller supplies every payload byte.  This module adds provenance metadata
and a checksum manifest, verifies the complete temporary directory, and only
then promotes it to the requested destination.  It intentionally contains no
SEC fetching, filing interpretation, model execution, or command-line code.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat as stat_module
from typing import Any
import uuid


ARTIFACT_SCHEMA_VERSION = "sec-audit-artifact-v1"
METADATA_FILENAME = "artifact_metadata.json"
CHECKSUMS_FILENAME = "checksums.json"

_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_CONTRACT_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_WINDOWS_DEVICE_RE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z", re.I)
_WINDOWS_FORBIDDEN = frozenset('<>:"|?*')
_RESERVED_NAMES = frozenset(
    {METADATA_FILENAME.casefold(), CHECKSUMS_FILENAME.casefold()}
)


if os.name == "nt":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_READ_ATTRIBUTES = 0x0080
    _SYNCHRONIZE = 0x00100000
    _DELETE = 0x00010000
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _CREATE_NEW = 1
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_DIRECTORY = 0x00000010
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    _FILE_BASIC_INFO_CLASS = 0
    _FILE_RENAME_INFO_CLASS = 3
    _FILE_DISPOSITION_INFO_CLASS = 4
    _FILE_ID_INFO_CLASS = 18

    class _FILE_ID_128(ctypes.Structure):
        _fields_ = [("Identifier", ctypes.c_ubyte * 16)]

    class _FILE_ID_INFO(ctypes.Structure):
        _fields_ = [
            ("VolumeSerialNumber", ctypes.c_ulonglong),
            ("FileId", _FILE_ID_128),
        ]

    class _FILE_BASIC_INFO(ctypes.Structure):
        _fields_ = [
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
        ]

    class _FILE_RENAME_INFO(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", wintypes.BOOLEAN),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    class _FILE_DISPOSITION_INFO(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

    _KERNEL32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _KERNEL32.CreateFileW.restype = wintypes.HANDLE
    _KERNEL32.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _KERNEL32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    _KERNEL32.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _KERNEL32.SetFileInformationByHandle.restype = wintypes.BOOL
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
    _KERNEL32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    _KERNEL32.GetDriveTypeW.restype = wintypes.UINT


@dataclass(frozen=True)
class _WindowsFileIdentity:
    volume_serial: int
    file_id: bytes


def _win_error(label: str) -> SecAuditArtifactError:
    code = ctypes.get_last_error()
    return SecAuditArtifactError(f"{label} failed with Windows error {code}")


def _win_close_handle(handle: int | None) -> None:
    if os.name == "nt" and handle not in (None, _INVALID_HANDLE_VALUE):
        _KERNEL32.CloseHandle(handle)


def _win_handle_attributes(handle: int) -> int:
    details = _FILE_BASIC_INFO()
    if not _KERNEL32.GetFileInformationByHandleEx(
        handle,
        _FILE_BASIC_INFO_CLASS,
        ctypes.byref(details),
        ctypes.sizeof(details),
    ):
        raise _win_error("artifact file-attribute query")
    return int(details.FileAttributes)


def _win_handle_identity(handle: int) -> _WindowsFileIdentity:
    details = _FILE_ID_INFO()
    if not _KERNEL32.GetFileInformationByHandleEx(
        handle,
        _FILE_ID_INFO_CLASS,
        ctypes.byref(details),
        ctypes.sizeof(details),
    ):
        raise _win_error("artifact file-identity query")
    return _WindowsFileIdentity(
        volume_serial=int(details.VolumeSerialNumber),
        file_id=bytes(details.FileId.Identifier),
    )


def _win_create_file(
    path: Path,
    *,
    access: int,
    share: int,
    disposition: int,
    flags: int,
) -> int:
    ctypes.set_last_error(0)
    handle = _KERNEL32.CreateFileW(
        str(path), access, share, None, disposition, flags, None
    )
    if handle in (None, _INVALID_HANDLE_VALUE):
        raise _win_error("artifact path open")
    return int(handle)


class _WindowsDirectoryLease(AbstractContextManager["_WindowsDirectoryLease"]):
    def __init__(
        self,
        path: Path,
        handle: int,
        identity: _WindowsFileIdentity,
        *,
        delete_access: bool,
        root_anchor: bool,
    ) -> None:
        self.path = path
        self.handle: int | None = handle
        self.identity = identity
        self.delete_access = delete_access
        self.root_anchor = root_anchor

    def __enter__(self) -> "_WindowsDirectoryLease":
        return self

    def close(self) -> None:
        handle, self.handle = self.handle, None
        _win_close_handle(handle)

    def __exit__(self, *args: Any) -> None:
        self.close()

    def assert_same(self) -> None:
        if self.handle is None or _win_handle_identity(self.handle) != self.identity:
            raise SecAuditArtifactError("artifact storage identity changed")
        share = _FILE_SHARE_READ
        if self.delete_access:
            share |= _FILE_SHARE_DELETE
        if self.root_anchor:
            share |= _FILE_SHARE_WRITE | _FILE_SHARE_DELETE
        reopened = _win_open_directory(
            self.path,
            delete_access=False,
            share=share,
            root_anchor=self.root_anchor,
        )
        try:
            if reopened.identity != self.identity:
                raise SecAuditArtifactError("artifact storage identity changed")
        finally:
            reopened.close()


def _win_open_directory(
    path: Path,
    *,
    delete_access: bool,
    share: int | None = None,
    root_anchor: bool = False,
) -> _WindowsDirectoryLease:
    access = _FILE_READ_ATTRIBUTES | _SYNCHRONIZE
    if not root_anchor:
        access |= _FILE_LIST_DIRECTORY
    if delete_access:
        access |= _DELETE
    if share is None:
        share = (
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE
            if root_anchor
            else _FILE_SHARE_READ
        )
    handle = _win_create_file(
        path,
        access=access,
        share=share,
        disposition=_OPEN_EXISTING,
        flags=_FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
    )
    try:
        attributes = _win_handle_attributes(handle)
        if not attributes & _FILE_ATTRIBUTE_DIRECTORY:
            raise SecAuditArtifactError("artifact storage component is not a directory")
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise SecAuditArtifactError(
                "artifact storage cannot contain links or reparse points"
            )
        return _WindowsDirectoryLease(
            path,
            handle,
            _win_handle_identity(handle),
            delete_access=delete_access,
            root_anchor=root_anchor,
        )
    except Exception:
        _win_close_handle(handle)
        raise


class _WindowsDirectoryChain(AbstractContextManager["_WindowsDirectoryChain"]):
    def __init__(self, leases: list[_WindowsDirectoryLease]) -> None:
        self.leases = leases

    @property
    def leaf(self) -> _WindowsDirectoryLease:
        return self.leases[-1]

    def __enter__(self) -> "_WindowsDirectoryChain":
        return self

    def assert_same(self) -> None:
        for lease in self.leases:
            lease.assert_same()

    def close(self) -> None:
        while self.leases:
            self.leases.pop().close()

    def __exit__(self, *args: Any) -> None:
        self.close()


def _win_pin_directory_chain(
    path: Path, *, create_missing: bool
) -> _WindowsDirectoryChain:
    absolute = Path(os.path.abspath(path))
    raw = str(absolute).replace("/", "\\")
    if raw.startswith("\\\\") or not absolute.anchor:
        raise SecAuditArtifactError("artifact storage must be on a local device")
    drive_type = int(_KERNEL32.GetDriveTypeW(absolute.anchor))
    if drive_type not in {2, 3, 5, 6}:
        raise SecAuditArtifactError("artifact storage must be on a local device")

    anchor = Path(absolute.anchor)
    leases: list[_WindowsDirectoryLease] = []
    try:
        leases.append(
            _win_open_directory(
                anchor, delete_access=False, root_anchor=True
            )
        )
        current = anchor
        for component in absolute.relative_to(anchor).parts:
            current = current / component
            if not _path_exists(current):
                if not create_missing:
                    raise SecAuditArtifactError(
                        "artifact directory does not exist or is not a real directory"
                    )
                current.mkdir(exist_ok=False)
            lease = _win_open_directory(current, delete_access=False)
            leases.append(lease)
            for pinned in leases:
                pinned.assert_same()
        return _WindowsDirectoryChain(leases)
    except Exception:
        while leases:
            leases.pop().close()
        raise


def _win_rename_open_directory(
    lease: _WindowsDirectoryLease, destination: Path
) -> None:
    if lease.handle is None or not lease.delete_access:
        raise SecAuditArtifactError("artifact temporary directory is not renameable")
    encoded = str(destination).encode("utf-16-le")
    offset = _FILE_RENAME_INFO.FileName.offset
    buffer = ctypes.create_string_buffer(
        ctypes.sizeof(_FILE_RENAME_INFO) + len(encoded)
    )
    info = _FILE_RENAME_INFO.from_buffer(buffer)
    info.ReplaceIfExists = False
    info.RootDirectory = None
    info.FileNameLength = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
    if not _KERNEL32.SetFileInformationByHandle(
        lease.handle,
        _FILE_RENAME_INFO_CLASS,
        buffer,
        len(buffer),
    ):
        raise _win_error("artifact promotion")
    lease.path = destination
    lease.assert_same()


def _win_mark_directory_delete(lease: _WindowsDirectoryLease) -> None:
    if lease.handle is None or not lease.delete_access:
        raise SecAuditArtifactError("artifact temporary directory is not deletable")
    info = _FILE_DISPOSITION_INFO(True)
    if not _KERNEL32.SetFileInformationByHandle(
        lease.handle,
        _FILE_DISPOSITION_INFO_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        raise _win_error("artifact temporary cleanup")


def _win_write_new_file(path: Path, payload: bytes) -> None:
    import msvcrt

    handle = _win_create_file(
        path,
        access=_GENERIC_WRITE | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        share=_FILE_SHARE_READ,
        disposition=_CREATE_NEW,
        flags=_FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
    )
    descriptor: int | None = None
    try:
        attributes = _win_handle_attributes(handle)
        if attributes & (_FILE_ATTRIBUTE_DIRECTORY | _FILE_ATTRIBUTE_REPARSE_POINT):
            raise SecAuditArtifactError("artifact output file is not a real file")
        descriptor = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
        handle = None
        with os.fdopen(descriptor, "wb", closefd=True) as file:
            descriptor = None
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)
        _win_close_handle(handle)


def _win_read_real_file(path: Path) -> tuple[bytes, _WindowsFileIdentity]:
    import msvcrt

    handle = _win_create_file(
        path,
        access=_GENERIC_READ | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        share=_FILE_SHARE_READ,
        disposition=_OPEN_EXISTING,
        flags=_FILE_FLAG_OPEN_REPARSE_POINT | _FILE_FLAG_SEQUENTIAL_SCAN,
    )
    descriptor: int | None = None
    try:
        attributes = _win_handle_attributes(handle)
        if attributes & (_FILE_ATTRIBUTE_DIRECTORY | _FILE_ATTRIBUTE_REPARSE_POINT):
            raise SecAuditArtifactError(
                "artifact must contain flat regular files only"
            )
        identity = _win_handle_identity(handle)
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        handle = None
        with os.fdopen(descriptor, "rb", closefd=True) as file:
            descriptor = None
            return file.read(), identity
    finally:
        if descriptor is not None:
            os.close(descriptor)
        _win_close_handle(handle)


def _win_cleanup_temporary(lease: _WindowsDirectoryLease) -> None:
    lease.assert_same()
    entries = list(os.scandir(lease.path))
    for entry in entries:
        child = Path(entry.path)
        details = child.lstat()
        reparse_flag = getattr(details, "st_file_attributes", 0) & 0x400
        if reparse_flag or not entry.is_file(follow_symlinks=False):
            raise SecAuditArtifactError(
                "unsafe entry prevented temporary artifact cleanup"
            )
        child.unlink()
    lease.assert_same()
    _win_mark_directory_delete(lease)


class SecAuditArtifactError(ValueError):
    """A sealed artifact or artifact request violated the storage contract."""


@dataclass(frozen=True)
class ArtifactVerification:
    """Verified identity and byte hashes for one sealed artifact directory."""

    artifact_dir: Path
    source_commit: str
    audit_contract_version: str
    file_checksums: tuple[tuple[str, str], ...]
    checksums_sha256: str

    @property
    def checksums(self) -> dict[str, str]:
        """Return a fresh filename-to-digest mapping."""

        return dict(self.file_checksums)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "source_commit": self.source_commit,
            "audit_contract_version": self.audit_contract_version,
            "checksums": self.checksums,
            "checksums_sha256": self.checksums_sha256,
        }


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_tagged(payload: bytes) -> str:
    return f"sha256:{_sha256_hex(payload)}"


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _normalize_source_commit(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("source_commit must be a Git commit string")
    normalized = value.strip().lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise SecAuditArtifactError(
            "source_commit must be a full 40- or 64-character hexadecimal commit"
        )
    return normalized


def _validate_contract_version(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("audit_contract_version must be text")
    if not _CONTRACT_VERSION_RE.fullmatch(value):
        raise SecAuditArtifactError(
            "audit_contract_version must be a safe non-empty version token"
        )
    return value


def _validate_flat_name(name: str, *, allow_generated: bool = False) -> str:
    if not isinstance(name, str):
        raise TypeError("artifact file names must be strings")
    if (
        not name
        or name in {".", ".."}
        or len(name.encode("utf-8")) > 255
        or any(ord(character) < 32 for character in name)
        or any(character in _WINDOWS_FORBIDDEN for character in name)
        or "/" in name
        or "\\" in name
        or name.endswith((" ", "."))
    ):
        raise SecAuditArtifactError(
            f"artifact file name must be one safe relative component: {name!r}"
        )

    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or len(posix.parts) != 1
        or len(windows.parts) != 1
    ):
        raise SecAuditArtifactError(
            f"artifact file name must be one safe relative component: {name!r}"
        )

    device_stem = name.split(".", 1)[0]
    if _WINDOWS_DEVICE_RE.fullmatch(device_stem):
        raise SecAuditArtifactError(f"artifact file name is reserved: {name!r}")

    folded = name.casefold()
    if folded in _RESERVED_NAMES:
        if not allow_generated or name not in {METADATA_FILENAME, CHECKSUMS_FILENAME}:
            raise SecAuditArtifactError(f"artifact file name is reserved: {name!r}")
    return name


def _normalize_files(files: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(files, Mapping):
        raise TypeError("files must be a mapping of relative names to bytes")
    normalized: dict[str, bytes] = {}
    casefolded: dict[str, str] = {}
    for name, payload in files.items():
        safe_name = _validate_flat_name(name)
        folded = safe_name.casefold()
        if folded in casefolded:
            raise SecAuditArtifactError(
                "artifact file names must also be unique when case-insensitive: "
                f"{casefolded[folded]!r}, {safe_name!r}"
            )
        if not isinstance(payload, bytes):
            raise TypeError(f"artifact payload for {safe_name!r} must be bytes")
        casefolded[folded] = safe_name
        normalized[safe_name] = payload
    return normalized


def _write_bytes(path: Path, payload: bytes) -> None:
    if os.name == "nt":
        _win_write_new_file(path, payload)
        return
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        # An unreadable path is not safe to treat as absent.
        return True


def _snapshot_entries(artifact_dir: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    casefolded: set[str] = set()
    for entry in artifact_dir.iterdir():
        _validate_flat_name(entry.name, allow_generated=True)
        folded = entry.name.casefold()
        if folded in casefolded:
            raise SecAuditArtifactError(
                "artifact contains case-insensitively duplicate file names"
            )
        casefolded.add(folded)
        if os.name == "nt":
            details = entry.lstat()
            reparse = getattr(details, "st_file_attributes", 0) & 0x400
            if reparse or not stat_module.S_ISREG(details.st_mode):
                raise SecAuditArtifactError(
                    "artifact must contain flat regular files only"
                )
            try:
                first, first_identity = _win_read_real_file(entry)
                second, second_identity = _win_read_real_file(entry)
            except SecAuditArtifactError as exc:
                raise SecAuditArtifactError(
                    "artifact must contain flat regular files only"
                ) from exc
            if first_identity != second_identity or first != second:
                raise SecAuditArtifactError("artifact file changed during verification")
            snapshot[entry.name] = first
        else:
            if entry.is_symlink() or not entry.is_file():
                raise SecAuditArtifactError(
                    "artifact must contain flat regular files only"
                )
            snapshot[entry.name] = entry.read_bytes()
    return snapshot


def _snapshot(
    artifact_dir: Path,
    *,
    directory_lease: _WindowsDirectoryLease | None = None,
) -> dict[str, bytes]:
    if os.name != "nt":
        if artifact_dir.is_symlink() or not artifact_dir.is_dir():
            raise SecAuditArtifactError(
                "artifact directory does not exist or is not a real directory: "
                f"{artifact_dir}"
            )
        return _snapshot_entries(artifact_dir)

    if directory_lease is not None:
        directory_lease.assert_same()
        first = _snapshot_entries(artifact_dir)
        directory_lease.assert_same()
        second = _snapshot_entries(artifact_dir)
        directory_lease.assert_same()
        if first != second:
            raise SecAuditArtifactError("artifact changed during verification")
        return first

    with _win_pin_directory_chain(
        artifact_dir, create_missing=False
    ) as pinned:
        pinned.assert_same()
        first = _snapshot_entries(artifact_dir)
        pinned.assert_same()
        second = _snapshot_entries(artifact_dir)
        pinned.assert_same()
        if first != second:
            raise SecAuditArtifactError("artifact changed during verification")
        return first


def _parse_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecAuditArtifactError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SecAuditArtifactError(f"{label} must be a JSON object")
    return value


def _verify_artifact(
    artifact_dir: str | os.PathLike[str],
    *,
    expected_checksums_sha256: str,
    expected_source_commit: str | None = None,
    expected_audit_contract_version: str | None = None,
    directory_lease: _WindowsDirectoryLease | None = None,
) -> ArtifactVerification:
    """Strictly verify one sealed flat-directory artifact.

    The checksum-manifest hash is mandatory external identity. Optional source
    values additionally bind the locally recorded provenance metadata.
    """

    directory = Path(artifact_dir).absolute()
    snapshot = _snapshot(directory, directory_lease=directory_lease)
    if METADATA_FILENAME not in snapshot or CHECKSUMS_FILENAME not in snapshot:
        raise SecAuditArtifactError(
            "artifact must contain artifact_metadata.json and checksums.json"
        )

    raw_checksums = snapshot[CHECKSUMS_FILENAME]
    checksums = _parse_json_object(raw_checksums, label=CHECKSUMS_FILENAME)
    if raw_checksums != _pretty_json_bytes(checksums):
        raise SecAuditArtifactError("checksums.json is not canonically serialized")

    for name, digest in checksums.items():
        _validate_flat_name(name, allow_generated=True)
        if name == CHECKSUMS_FILENAME:
            raise SecAuditArtifactError("checksums.json cannot checksum itself")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise SecAuditArtifactError(
                f"checksum for {name!r} must be lowercase SHA-256"
            )

    actual_names = set(snapshot) - {CHECKSUMS_FILENAME}
    if set(checksums) != actual_names:
        raise SecAuditArtifactError("checksum manifest file set is not exact")
    for name in sorted(actual_names):
        observed = _sha256_hex(snapshot[name])
        if not hmac.compare_digest(observed, checksums[name]):
            raise SecAuditArtifactError(f"artifact checksum mismatch: {name}")

    raw_metadata = snapshot[METADATA_FILENAME]
    metadata = _parse_json_object(raw_metadata, label=METADATA_FILENAME)
    if raw_metadata != _pretty_json_bytes(metadata):
        raise SecAuditArtifactError(
            "artifact_metadata.json is not canonically serialized"
        )
    if set(metadata) != {
        "artifact_schema_version",
        "audit_contract_version",
        "source_commit",
    }:
        raise SecAuditArtifactError("artifact metadata schema is not exact")
    if metadata["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise SecAuditArtifactError("artifact schema version is unsupported")
    source_commit = _normalize_source_commit(metadata["source_commit"])
    contract_version = _validate_contract_version(metadata["audit_contract_version"])
    if metadata["source_commit"] != source_commit:
        raise SecAuditArtifactError("source_commit must use canonical lowercase form")

    if expected_source_commit is not None:
        expected_commit = _normalize_source_commit(expected_source_commit)
        if not hmac.compare_digest(source_commit, expected_commit):
            raise SecAuditArtifactError("artifact source commit does not match expected")
    if expected_audit_contract_version is not None:
        expected_contract = _validate_contract_version(
            expected_audit_contract_version
        )
        if contract_version != expected_contract:
            raise SecAuditArtifactError(
                "artifact audit contract version does not match expected"
            )

    checksums_sha256 = _sha256_tagged(raw_checksums)
    if (
        not isinstance(expected_checksums_sha256, str)
        or not _TAGGED_SHA256_RE.fullmatch(expected_checksums_sha256)
    ):
        raise SecAuditArtifactError(
            "expected_checksums_sha256 must be tagged lowercase SHA-256"
        )
    if not hmac.compare_digest(checksums_sha256, expected_checksums_sha256):
        raise SecAuditArtifactError("checksum manifest does not match expected")

    return ArtifactVerification(
        artifact_dir=directory,
        source_commit=source_commit,
        audit_contract_version=contract_version,
        file_checksums=tuple(
            sorted((str(name), str(digest)) for name, digest in checksums.items())
        ),
        checksums_sha256=checksums_sha256,
    )


def verify_artifact(
    artifact_dir: str | os.PathLike[str],
    *,
    expected_checksums_sha256: str,
    expected_source_commit: str | None = None,
    expected_audit_contract_version: str | None = None,
) -> ArtifactVerification:
    """Strictly verify one sealed flat-directory artifact."""

    return _verify_artifact(
        artifact_dir,
        expected_checksums_sha256=expected_checksums_sha256,
        expected_source_commit=expected_source_commit,
        expected_audit_contract_version=expected_audit_contract_version,
    )


def read_artifact_file(
    artifact_dir: str | os.PathLike[str], name: str
) -> bytes:
    """Read one flat artifact file without following directory or file links."""

    safe_name = _validate_flat_name(name, allow_generated=True)
    directory = Path(artifact_dir).absolute()
    path = directory / safe_name
    if os.name == "nt":
        with _win_pin_directory_chain(
            directory, create_missing=False
        ) as pinned:
            pinned.assert_same()
            payload, _ = _win_read_real_file(path)
            pinned.assert_same()
            return payload
    if directory.is_symlink() or not directory.is_dir():
        raise SecAuditArtifactError(
            "artifact directory does not exist or is not a real directory"
        )
    if path.is_symlink() or not path.is_file():
        raise SecAuditArtifactError("artifact file is not a real regular file")
    return path.read_bytes()


def _seal_artifact_windows(
    destination: Path,
    all_files: Mapping[str, bytes],
    *,
    normalized_commit: str,
    contract_version: str,
) -> ArtifactVerification:
    temporary = destination.parent / (
        f".{destination.name}.{uuid.uuid4().hex}.sealing"
    )
    with _win_pin_directory_chain(
        destination.parent, create_missing=True
    ) as parent_chain:
        parent_chain.assert_same()
        if _path_exists(destination):
            raise SecAuditArtifactError(f"artifact already exists: {destination}")

        temporary_lease: _WindowsDirectoryLease | None = None
        promoted = False
        try:
            temporary.mkdir(exist_ok=False)
            temporary_lease = _win_open_directory(
                temporary, delete_access=True
            )
            parent_chain.assert_same()
            temporary_lease.assert_same()
            for name in sorted(all_files):
                parent_chain.assert_same()
                temporary_lease.assert_same()
                _write_bytes(temporary / name, all_files[name])
                temporary_lease.assert_same()
            checksums = {
                name: _sha256_hex(all_files[name]) for name in sorted(all_files)
            }
            checksums_payload = _pretty_json_bytes(checksums)
            _write_bytes(temporary / CHECKSUMS_FILENAME, checksums_payload)
            expected_checksums_sha256 = _sha256_tagged(checksums_payload)
            _verify_artifact(
                temporary,
                expected_checksums_sha256=expected_checksums_sha256,
                expected_source_commit=normalized_commit,
                expected_audit_contract_version=contract_version,
                directory_lease=temporary_lease,
            )
            parent_chain.assert_same()
            temporary_lease.assert_same()
            if _path_exists(destination):
                raise SecAuditArtifactError(
                    f"artifact already exists: {destination}"
                )
            # The open temporary-directory handle pins the complete ancestor
            # chain on Windows. Release the read-only parent leases so the
            # handle-based rename may update the parent directory entry.
            parent_chain.close()
            _win_rename_open_directory(temporary_lease, destination)
            promoted = True
            temporary_lease.assert_same()
            return _verify_artifact(
                destination,
                expected_checksums_sha256=expected_checksums_sha256,
                expected_source_commit=normalized_commit,
                expected_audit_contract_version=contract_version,
                directory_lease=temporary_lease,
            )
        except Exception:
            if temporary_lease is not None and not promoted:
                try:
                    _win_cleanup_temporary(temporary_lease)
                except Exception:
                    # Leave an identity-pinned temporary directory behind rather
                    # than applying pathname cleanup after unexpected interference.
                    pass
            raise
        finally:
            if temporary_lease is not None:
                temporary_lease.close()


def seal_artifact(
    artifact_dir: str | os.PathLike[str],
    files: Mapping[str, bytes],
    *,
    source_commit: str,
    audit_contract_version: str,
) -> ArtifactVerification:
    """Create, verify, and atomically promote a deterministic audit artifact."""

    payloads = _normalize_files(files)
    normalized_commit = _normalize_source_commit(source_commit)
    contract_version = _validate_contract_version(audit_contract_version)
    destination = Path(artifact_dir).absolute()

    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "audit_contract_version": contract_version,
        "source_commit": normalized_commit,
    }
    all_files = {
        **payloads,
        METADATA_FILENAME: _pretty_json_bytes(metadata),
    }
    if os.name == "nt":
        return _seal_artifact_windows(
            destination,
            all_files,
            normalized_commit=normalized_commit,
            contract_version=contract_version,
        )

    if _path_exists(destination):
        raise SecAuditArtifactError(f"artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{uuid.uuid4().hex}.sealing"
    )

    try:
        temporary.mkdir(exist_ok=False)
        for name in sorted(all_files):
            _write_bytes(temporary / name, all_files[name])
        checksums = {
            name: _sha256_hex(all_files[name]) for name in sorted(all_files)
        }
        _write_bytes(temporary / CHECKSUMS_FILENAME, _pretty_json_bytes(checksums))
        expected_checksums_sha256 = _sha256_tagged(
            _pretty_json_bytes(checksums)
        )
        verification = verify_artifact(
            temporary,
            expected_checksums_sha256=expected_checksums_sha256,
            expected_source_commit=normalized_commit,
            expected_audit_contract_version=contract_version,
        )
        if _path_exists(destination):
            raise SecAuditArtifactError(f"artifact already exists: {destination}")
        temporary.replace(destination)
    except Exception:
        if _path_exists(temporary):
            if temporary.is_dir() and not temporary.is_symlink():
                shutil.rmtree(temporary)
            else:  # pragma: no cover - defensive against external interference
                temporary.unlink()
        raise

    return ArtifactVerification(
        artifact_dir=destination,
        source_commit=verification.source_commit,
        audit_contract_version=verification.audit_contract_version,
        file_checksums=verification.file_checksums,
        checksums_sha256=verification.checksums_sha256,
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactVerification",
    "CHECKSUMS_FILENAME",
    "METADATA_FILENAME",
    "SecAuditArtifactError",
    "read_artifact_file",
    "seal_artifact",
    "verify_artifact",
]
