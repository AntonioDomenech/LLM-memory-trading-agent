"""Deterministic, filesystem-only sealing for flat SEC audit artifacts.

The caller supplies every payload byte.  This module adds provenance metadata
and a checksum manifest, verifies the complete temporary directory, and only
then promotes it to the requested destination.  It intentionally contains no
SEC fetching, filing interpretation, model execution, or command-line code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
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
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _snapshot(artifact_dir: Path) -> dict[str, bytes]:
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise SecAuditArtifactError(
            f"artifact directory does not exist or is not a real directory: {artifact_dir}"
        )

    snapshot: dict[str, bytes] = {}
    casefolded: set[str] = set()
    for entry in artifact_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise SecAuditArtifactError("artifact must contain flat regular files only")
        _validate_flat_name(entry.name, allow_generated=True)
        folded = entry.name.casefold()
        if folded in casefolded:
            raise SecAuditArtifactError(
                "artifact contains case-insensitively duplicate file names"
            )
        casefolded.add(folded)
        snapshot[entry.name] = entry.read_bytes()
    return snapshot


def _parse_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecAuditArtifactError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SecAuditArtifactError(f"{label} must be a JSON object")
    return value


def verify_artifact(
    artifact_dir: str | os.PathLike[str],
    *,
    expected_checksums_sha256: str,
    expected_source_commit: str | None = None,
    expected_audit_contract_version: str | None = None,
) -> ArtifactVerification:
    """Strictly verify one sealed flat-directory artifact.

    The checksum-manifest hash is mandatory external identity. Optional source
    values additionally bind the locally recorded provenance metadata.
    """

    directory = Path(artifact_dir).absolute()
    snapshot = _snapshot(directory)
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
    if _path_exists(destination):
        raise SecAuditArtifactError(f"artifact already exists: {destination}")

    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "audit_contract_version": contract_version,
        "source_commit": normalized_commit,
    }
    all_files = {
        **payloads,
        METADATA_FILENAME: _pretty_json_bytes(metadata),
    }
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
    "seal_artifact",
    "verify_artifact",
]
